import datetime
import io
import math
import os
import pickle
import sys

import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

def harmonicrange(start, stop, splits):
	"""
	Generator for spliting a range(@start,@stop) by number of equal intervals @splits.
	This ignores the start and end of the range.
	For example,
		harmonicrange(10, 20, 2) yields [15]
		harmonicrange(10, 20, 3) yields [13,16]

	Unused remainder is given to the last interval. For the larger numbers involved here (RR intervals of 350) with typical splits of 2 or 3, the remainder is irrelevant.
	Perhaps would be more "ideal" to split the remainder over all the intervals but for my purpose it doesn't matter.
	"""

	# Splitting (stop-start) results in @k between splits with a remainder of @m
	k,m = divmod(stop-start, splits)
	# If @m is not subtracted off, then the last yield from range() will be @m away from @stop, which is not desired
	for idx in range(start, stop-m, k):
		# Skip the first and last index
		if idx == start or idx == stop: continue
		yield idx

def GenHilbert(N):
	"""Generate a discrete Hilbert transform filter."""
	if not N%2:
		raise ValueError("Hilbert transform must be odd length")

	ret = []
	for i in range( -int((N-1)/2), int((N+1)/2) ):
		if i != 0:
			_ = i*math.pi/2.0
			ret.append( math.sin(_)*math.sin(_)/_ )
		else:
			ret.append(0)

	return ret

def ApplyFilter(x, b):
	ret = scipy.signal.filtfilt(np.array(b), np.array([1]), np.array(x))
	return ret

def binarysearchlist(lst, r):
	"""
	Quickly search list @lst for an item within the range @r.
	Returns the index of the matched item.
	"""

	# Nonthing to find
	if len(lst) == 0:
		return None

	a = 0
	b = len(lst)-1
	for cnt in range(100):
		# Not found in the list
		if a == b:
			return None

		# Just finish with a linear search
		if b-a < 25:
			for i in range(a,b+1):
				if lst[i] in r:
					return i
			else:
				# Not found
				return None

		# Look half way
		idx = round( (b-a)/2+a )
		x = lst[idx]

		# Found an end point, stop here
		if x in r:
			return idx

		# Reaching this point means that x in not in the range, so must be higher or lower
		if x < r.start:
			# Search upper half
			a = idx
		else:
			# Search lower half
			b = idx

	else:
		# Avoid infinite loops
		cnt += 1
		if cnt > 100:
			raise NotImplementedError("Loop took too long: len(lst)=%d, a=%d, b=%d, r=%s, idx=%d, x=%d" % (len(lst), a, b, r, idx, x))

def findinrange(lst, r):
	"""
	Find all items in list @lst within range @r.
	Assumes @r is sorted so a binary search can be done for performance reasons.
	Returns empty list if no items around found
	"""

	start = binarysearchlist(lst, r)

	# No items found in range
	if start is None:
		return []

	# Start with found indexed item
	ret = [lst[start]]

	# Step down until start of the range
	for i in range(start-1, -1, -1):
		v = lst[i]
		if v >= r.start:
			ret.insert(0, v)
		else:
			break

	# Step up until end of the range
	len_lst = len(lst)
	for i in range(start, len_lst):
		v = lst[i]
		if v <= r.stop:
			ret.append(v)
		else:
			break

	return ret

class pyzestyecg:
	def __init__(self, wiff, params=None):
		default = __class__.GetDefaultParams()

		# Copy defaults and update if provided
		a = dict(default)
		if params is not None:
			a.update(params)

		self._wiff = wiff
		self._params = a

	@staticmethod
	def GetDefaultParams():
		return {
			'HilbertLength': 49,
			'Smoothing1': 40,
			'Smoothing2': 20,
			'WindowWidthMax': 35,
			'MaximumWindowWidth': 60,
			'RatioCutoff': 0.05,
			'Cutoffs': {
				'varPercentile': {
					'minimum': (1000.0, -100),
					'desired': (5000.0, +1),
				},
				'Points': 3,
				'Harmonics': {
					'Low': 0.85,
					'High': 1.15,
					'MaxMultiple': 4,
					'LeadCorrelates': 3,
				}
			},
			'LeadCorrelateWindow': 10,
			'PNG': {
				'speed': 200,
				'width': 20,
			},
		}

	@property
	def WIFF(self):
		return self._wiff

	@property
	def Params(self):
		return self._params

	def GetPotentialsAndPeaks(self, ignore, noise):
		"""
		Get the potential peaks and filtered peaks from the raw data.
		Use the @ignore list of (start,stop) tuples of regions to ignore and not analyze.
		"""

		dat = None
		i = None
		chans = None

		potentials = None
		peaks = None

		# Convert to ranges, easier to use logic against it
		rignores = [range(_[0],_[1]) for _ in ignore]
		rnoise = [range(_[0],_[1]) for _ in noise]

		# Iterate over the data and put into arrays
		print(['load', datetime.datetime.utcnow()])
		for fidx,chans,samps in self.WIFF.recording[1].GetAllFrames():
			if dat is None:
				# Avoid doing this every iteration since it doesn't change
				i = len(chans)
				dat = [[] for _ in range(i)]
				potentials = {_:[] for _ in chans}
				peaks = {_:[] for _ in chans}

			for n in range(i):
				dat[n].append(samps[n])

		# Data loaded, now process
		for cidx,cname in enumerate(chans):
			potentials[cname], peaks[cname] = self.processecg_single(cname, dat[cidx], self.Params, rignores, rnoise)

		return (potentials, peaks)

	def GetCorrelate(self, chans, potentials, peaks):
		"""
		Correlated peaks across leads (they don't always happen at the same time index).
		"""

		keys = {}
		correlate = {}
		for cname in chans:
			keys[cname] = sorted(peaks[cname].keys())

		winwidth = self.Params['LeadCorrelateWindow']
		for mainkey in chans:
			print(['A', mainkey, datetime.datetime.utcnow()])
			correlate[mainkey] = {}

			for k in keys[mainkey]:
				r = range(k-winwidth, k+winwidth+1)

				correlate[mainkey][k] = []

				for cname in chans:
					if cname == mainkey: continue

					correlate[mainkey][k].append( findinrange(keys[cname], r) )

		return correlate

	def GetKeepKeys(self, chans, peaks):
		keep_keys = {k:[] for k in peaks.keys()}
		points = {k:{} for k in peaks.keys()}

		# 4)
		# NOTE: This is currently just empirically determined, better if some sort of non-linear model (ie, "AI") were used instead
		for cname in peaks.keys():
			print(['A', cname, datetime.datetime.utcnow()])
			for k in sorted(peaks[cname].keys()):
				v = peaks[cname][k]

				p = __class__.scoreit(self.Params, cname, k, v)
				points[cname][k] = p

				if sum(p) >= self.Params['Cutoffs']['Points']:
					keep_keys[cname].append(k)
				else:
					pass

		return (points, keep_keys)

	def GetRemoveKeys(self, chans, potentials, peaks, correlate, points, keep):
		remove = []
		for cname,v in keep.items():
			print(['A', cname, datetime.datetime.utcnow()])
			for k in v:
				# Get correlated indices and current points for this peak
				corr = correlate[cname][k]
				corr_len = [len(_) > 0 for _ in corr].count(True)
				p = points[cname][k]

				# Iterate thrugh other leads and check how
				idx = 0
				cnt = 0
				for n in chans:
					if n == cname: continue

					# Check that any of the peaks correlated for each lead scored enough points to be kept
					if any([_ for _ in corr[idx] if _ in keep[n]]):
						cnt += 1

					idx += 1

				# This peak only matches 0-2 other leads, probably not enough to be a QRS so get rid of it
				if cnt < 3:
					remove.append( (cname,k) )

		return remove

	def CheckUserFilter(self, chans, points, keep, remove, user):
		"""
		Check that the user list of points @user is valid.
		"""

		for cname in keep.keys():
			user['Keep'][cname] = []
			user['Remove'][cname] = []

	def CalculateRR(self, chans, peaks, correlate, keep, remove, user, final, intervals, noise):
		"""
		Calculate RR based on the intervals provided using an incremental histogram.
		@intervals is a dictionary mapping named regions to a list of (start,stop) frames that are in that named region.
		@noise is a list of (start,stop) regions that are ignored from analysis as user indicates it is noise.
		"""

		# Final list of peaks
		#  Those in @keep are kept
		#  Unless in @remove or @user['Remove']
		#  Add [back in] any points in @user['Keep']
		final = {}

		# 6A)
		# 6B)
		print(['A', datetime.datetime.utcnow()])
		histo = {}
		for cname in keep.keys():
			sub = []
			for v in keep[cname]:
				if (cname,v) in remove:
					pass
				elif v in user['Remove'][cname]:
					pass
				else:
					sub.append(v)

			sub += user['Keep'][cname]
			sub = list(set(sub))
			sub.sort()
			# Store final list
			final[cname] = sub


		return final

	def CalculateRR_whole(self, chans, peaks, correlate, keep, remove, user, final, intervals, noise):
		"""
		Calculate RR based on the intervals provided using a histogram of the complete recording.
		@intervals is a dictionary mapping named regions to a list of (start,stop) frames that are in that named region.
		@noise is a list of (start,stop) regions that are ignored from analysis as user indicates it is noise.
		"""

		# Final list of peaks
		#  Those in @keep are kept
		#  Unless in @remove or @user['Remove']
		#  Add [back in] any points in @user['Keep']
		final = {}

		# 6A)
		# 6B)
		print(['A', datetime.datetime.utcnow()])
		histo = {}
		for cname in keep.keys():
			# Copy entire list first then remove
			final[cname] = {k:{} for k in keep[cname]}

			for k in keep[cname]:
				if (cname,k) in remove:
					final[cname][k]['poorlycorrelated'] = True
				else:
					final[cname][k]['poorlycorrelated'] = False

			sub = list(final[cname].keys())
			sub.sort()
			len_sub = len(final[cname])
			for i in range(1,len_sub):
				rr = sub[i] - sub[i-1]
				if rr not in histo: histo[rr] = 0
				histo[rr] += 1

		for k in sorted(histo.keys()):
			print("%20s %s" % (str(k), histo[k]))

		print(['B', datetime.datetime.utcnow()])

		# 6C)
		# The fundamental frequency of a dataset with harmonics necessarily is less than mean of the dataset
		# (If the dataset contained only a single harmonic then the mean is the fundamental frequency)
		# So the fundamental frequency is limited to min(histo.keys()) to avg
		print(['C', datetime.datetime.utcnow()])
		avg = sum([k*v for k,v in histo.items()])/sum(histo.values())

		# 6D)
		# Get the average of the first harmonic, this average is likely pretty close
		print(['D', datetime.datetime.utcnow()])
		f0 = 0
		len_f0 = 0
		avg2 = int(avg)+1
		for k,v in histo.items():
			if k > avg2: continue

			f0 += k*v
			len_f0 += v
		f0 /= len_f0

		# 6E)
		# Now go back through ECG data and find gaps larger than f0, particularly if an integer multiple suggesting missed complexes
		print(['E', datetime.datetime.utcnow()])
		too_soon = {k:[] for k in chans}
		for cname in chans:
			print(['E1', cname, datetime.datetime.utcnow()])
			last_k = None
			for k in sorted(final[cname].keys()):
				# Skip first
				if last_k is None:
					last_k = k
					final[cname][k]['f0'] = f0
					final[cname][k]['h'] = 1
					final[cname][k]['mult'] = 1
					final[cname][k]['result'] = 'initialpoint'
					final[cname][k]['toosoon'] = False
					continue

				h = (k-last_k)/f0
				mult = round(h)
				final[cname][k]['f0'] = f0
				final[cname][k]['h'] = h
				final[cname][k]['mult'] = mult
				final[cname][k]['result'] = None
				final[cname][k]['toosoon'] = False

				if last_k in [_[0] for _ in too_soon[cname]]:
					last_k = k
					continue

				if h < self.Params['Cutoffs']['Harmonics']['Low']:
					# Weird point, probably PAC/PVC or P wave
					too_soon[cname].append( (k, last_k, h, mult) )
					final[cname][k]['result'] = 'toosoon'
					pass
				elif h < self.Params['Cutoffs']['Harmonics']['High']:
					# First harmonic, skip it
					final[cname][k]['result'] = '1st'
					pass
				elif mult > self.Params['Cutoffs']['Harmonics']['MaxMultiple']:
					# Too many harmonics high
					final[cname][k]['result'] = 'toomanyharmonics'
					pass
				else:
					final[cname][k]['result'] = 'harmonic'
					for idx in harmonicrange(last_k, k, mult):
						toadd = __class__.findmissingharmonicpeak(peaks, final, correlate, cname, last_k, k, idx, self.Params, chans)
						for idx in toadd:
							final[cname][k]['f0'] = f0
							final[cname][k]['h'] = (idx-k)/f0
							final[cname][k]['mult'] = round(final[cname][k]['h'])
							final[cname][k]['result'] = 'correlated'
							final[cname][k]['toosoon'] = False
						# Continue in case there's more harmonics found

				# Update index for next loop
				last_k = k

		# 6F)
		# TODO: for now just remove them by flagging them
		print(['F', datetime.datetime.utcnow()])
		for cname in too_soon.keys():
			for k, last_k, h, mult in too_soon[cname]:
				final[cname][k]['toosoon'] = True

		return final

	def ExportPeaksByPNG(self, chans, potentials, peaks, correlate, keep, remove, user, final, intervals, noise, filegenerator, filesaver_peaks, filesaver_RR, width=10, speed=100):
		"""
		Export peak data that goes with each PNG file.
		Lot of copy/paste logic from ExportPNG()
		"""

		### COPY/pASTE FROM EXPORTPNG ###
		tstart = self.WIFF.recording[1].frame_table.fidx_start
		tend = self.WIFF.recording[1].frame_table.fidx_end

		# Total time delta across all leads
		delta = tend - tstart

		# Get sampling rate
		freq = self.WIFF.recording[1].sampling

		# Calculate samples per page with 25.4 mm = 1 inch
		# samps = (in * 25.4 mm/in) / (mm/sec) * (samples/sec)
		samps = (width * 25.4) / speed * freq

		# Truncate to whole sample to step
		step = int(samps)
		r = range(0, delta, step)
		### COPY/pASTE FROM EXPORTPNG ###

		# Save a channels list
		with filegenerator(0) as f:
			dat = '\n'.join(chans)
			dat = dat.encode('utf8')
			f.write(dat)
			filesaver_RR('channels', f)

		for idx,fidx in enumerate(r):
			rseg = range(fidx, fidx+step)

			for cname in chans:
				# Filter out the peaks data for the time slice
				potentials_points = {k:v for k,v in potentials[cname].items() if k in rseg}
				peaks_points = {k:v for k,v in peaks[cname].items() if k in rseg}
				keep_points = [_ for _ in keep[cname] if _ in rseg]
				remove_points = [_ for _ in remove if _[0] == cname and _[1] in rseg]
				userkeep_points = [_ for _ in user['Keep'][cname] if _ in rseg]
				userremove_points = [_ for _ in user['Remove'][cname] if _ in rseg]
				final_points = {k:v for k,v in final[cname].items() if k in rseg}

				with filegenerator(idx) as f:
					header = "Start\t%d\nEnd\t%d\nStep\t%d\nSamplingRate\t%d\nLead\t%s\n" % (fidx, fidx+step, step, freq, cname)
					header += "\t".join(['Index','Time','Keep','Remove','UserKeep','UserRemove','Ignored','Noisy','Window','pre','post','sum','ratio','sum/mean','MaximumIndex','MaximumPercentile'])
					header += '\n'
					header += "\t".join(['Index','Time','len(preRank)','avg(preRank)','len(sum/mean%max)','avg(sum/mean%max)','peakPercentile','varPercentile','Score','sum(score)','Potentials'])
					header += '\n'
					header += "\t".join(['Index','Time','f0','h','mult','result','toosoon','poorlycorrelated'])
					header += '\n\n'
					f.write(header.encode('utf8'))

					dat = []
					for k,v in potentials_points.items():
						d = (
							k,
							k/freq,
							int(k in keep_points), # 1 if a keep point
							int(k in remove_points), # 1 if a remove point
							int(k in userkeep_points), # 1 if a user kept point
							int(k in userremove_points), # 1 if a user removed point
							int(v['ignored']), # 1 if in an ignore interval
							int(v['noisy']), # 1 if in a noise interval
							v['window'],
							v['pre'],
							v['post'],
							v['sum'],
							v['ratio'],
							v['sum/mean'],
							v.get('Maximum', 0) and v['Maximum']['idx'] or 0,
							v.get('Maximum', 0) and v['Maximum']['Percentile'] or 0,
						)
						dat.append(d)
					dat.sort(key=lambda _:_[0])

					for d in dat:
						z = "\t".join([str(_) for _ in d]) + '\n'
						f.write( z.encode('utf8'))

					# Space rows
					f.write('\n'.encode('utf8'))

					dat.clear()
					for k,v in peaks_points.items():
						score = __class__.scoreit(self.Params, cname,k,v)
						d = (
							k,
							k/freq,
							len(v['preRank']),
							sum(v['preRank'])/len(v['preRank']),
							len(v['sum/mean%max']),
							sum(v['sum/mean%max'])/len(v['sum/mean%max']),
							v['peakPercentile'],
							v['varPercentile'],
							",".join(map(str,score)),
							sum(score),
							",".join([str(_) for _ in v['potentials']]),
						)
						dat.append(d)
					dat.sort(key=lambda _:_[0])

					for d in dat:
						z = "\t".join([str(_) for _ in d]) + '\n'
						f.write( z.encode('utf8'))

					# Space rows
					f.write('\n'.encode('utf8'))

					# Write final points
					dat.clear()
					for k,v in final_points.items():
						d = (
							k,
							k/freq,
							v['f0'],
							v['h'],
							v['mult'],
							v['result'],
							int(v['toosoon']),
							int(v['poorlycorrelated']),
						)
						dat.append(d)
					dat.sort(key=lambda _:_[0])

					for d in dat:
						z = "\t".join([str(_) for _ in d]) + '\n'
						f.write( z.encode('utf8'))

					filesaver_peaks(idx,cname, f)

	def ExportRR(self, chans, peaks, correlate, keep, remove, user, final, intervals, noise, filegenerator, filesaver):
		"""
		Export RR data as three columns: start index, end index, RR interval (end-start)
		One file per lead in the EKG.
		"""

		# Map interval name to the RR's within the range
		int_rrs = {}

		# Map ranges to the analysis interval
		int_map = {}
		for k,v in intervals.items():
			km = range(v[0], v[1])
			int_map[km] = k
			int_rrs[k] = {cname:[] for cname in chans}

		# Save all RR's for each lead (include start,stop times)
		# Each row provides the start and end frame in the WIFFECG file
		# The DELTA is the difference between them (ie, the RR interval)
		# Additionally, any named intervals that include the RR interval in its entirely is listed after the delta
		# There may be none, one, or many additional columns of interval names
		#   START END DELTA INTERVAL1 INTERVAL2 ...
		for cname in chans:
			with filegenerator(cname) as f:
				# Get all keys on this lead
				keys = list(final[cname].keys())

				# Start iterating from [1] with previous index [0] and then increment from there
				last_idx = 0
				for idx in range(1,len(keys)):
					print([last_idx, idx, len(keys)])
					s = keys[last_idx]
					e = keys[idx]
					d = e-s

					# Skip these (note that last_idx doesn't change so RR should be correct)
					if final[cname][e]['toosoon']:
						continue

					# Include RR if it starts and ends within the interval time (cannot overlap with start/end outside the range)
					names = []
					for r in int_map.keys():
						if s in r and e in r:
							k = int_map[r]
							int_rrs[k][cname].append(d)
							names.append(k)

					# Save for this lead: START END DELTA and any named intervals
					z = "%d\t%d\t%d" % (s,e,d)
					names = list(set(names))
					if len(names):
						names.sort()
						z += '\t' + '\t'.join(names)
					a += '\n'

					f.write(z.encode('utf8'))
					last_idx = idx

				filesaver(cname,f)

		# Save all RR data for each interval
		#   Lead I	RR1	RR2	RR3	RR4	...
		#   Lead II	RR1	RR2	RR3	RR4	...
		#   ...
		for k in int_rrs.keys():
			with filegenerator("named-%s"%k) as f:

				# With one lead per row with first column being the lead name
				for cname in int_rrs[k]:
					# Write lead name first
					# If the lead has no intervals mapped to it, then it would be a simple line of "NAME\n" without any data
					f.write(cname.encode('utf8'))

					# Write each value as a tab-delimited column
					for v in int_rrs[k][cname]:
						f.write(("\t%d"%v).encode('utf8'))
					f.write('\n'.encode('utf8'))

				filesaver('named-%s'%k, f)

	def ExportPNG(self, peaks, filegenerator, filesaver, width=10, speed=100):
		"""
		Export data to PNG files.
		@filegenerator -- Function that returns an object to write() to, only argument is 
		@filesaver -- Function that takes the object returned from @filegenerator and saves it
		@width -- width of image in inches
		@speed -- speed of ECG (standard 12 lead is 25 mm/sec, default to 100)
		"""

		# 6 leads so 6 vertical sub plots
		# FIXME: calculate programmatically
		len_chans = 6

		fig,axs = plt.subplots(len_chans)

		tstart = self.WIFF.recording[1].frame_table.fidx_start
		tend = self.WIFF.recording[1].frame_table.fidx_end

		# Total time delta across all leads
		delta = tend - tstart

		# Get sampling rate
		freq = self.WIFF.recording[1].sampling

		# Calculate samples per page with 25.4 mm = 1 inch
		# samps = (in * 25.4 mm/in) / (mm/sec) * (samples/sec)
		samps = (width * 25.4) / speed * freq

		# Truncate to whole sample to step
		step = int(samps)
		r = range(0, delta, step)

		for i in range(0,6):
			axs[i].get_xaxis().set_visible(False)
		axs[-1].get_xaxis().set_visible(True)

		for idx,fidx in enumerate(r):
			print(['page', datetime.datetime.utcnow(), (idx, len(r)), (fidx, tend, tend-fidx)])
			# idx -- File index
			# fidx -- Frame index (start) of the section to be saved to a PNG

			s = slice(fidx, fidx+step)
			rseg = range(fidx, fidx+step)

			# Get data for this image section
			d = []
			chans = None
			for fidxstart,chans,dat in self.WIFF.recording[1].GetSliceFrames(s):
				d.append(dat)

			# Iterate over each lead in the time slice
			for i in range(0,len(d[0])):
				cname = chans[i]

				# Clear the subfigure
				axs[i].cla()

				# Apply lead label to the y axis
				axs[i].set_ylabel(cname)

				# pull out the column of data to plot
				subd = [_[i] for _ in d]
				axs[i].plot(list(map(lambda _:(_+idx*step)/freq, range(0, len(d)))), subd)

				# Filter out the peaks data for the time slice
				match_points = [_ for _ in peaks[cname] if _ in rseg and not peaks[cname][_]['toosoon']]

				# Plot as red circles ("ro") on top of the ECG data
				#print([len(subd), len(d), fidx, match_points])
				#print([_-fidx for _ in match_points])
				axs[i].plot([_/freq for _ in match_points], [subd[_-fidx] for _ in match_points], 'ro')

			# Labels
			plt.xlabel("Time (sec)")
			axs[-1].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
			axs[-1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
			axs[-1].xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.025))

			# Format figure size
			fig.set_figwidth(width)
			fig.set_figheight(12)

			with filegenerator(idx) as f:
				plt.savefig(f, format='png', bbox_inches='tight')

				f.seek(0)
				filesaver(idx, f)

	@staticmethod
	def findmissingharmonicpeak(peaks, final, correlate, cname, last_k, k, idx, params, header):
		"""
		Try to find a peak for lead @cname around index @idx, which was found using harmonic analysis between indices @last_k and @k.
		"""
		toadd = []

		winwidth = params['LeadCorrelateWindow']
		r = range(idx - winwidth, idx + winwidth)

		#print(['find peak', cname, idx, r])
		possibles = [_ for _ in peaks[cname].keys() if _ in r]
		#print(['possibles', possibles])

		for p in possibles:
			score = __class__.scoreit(params, cname,p, peaks[cname][p])
			print([p, peaks[cname][p], score])

			cnt = 0
			idx = 0
			for cn in header[1:]:
				if cn == cname: continue

				for hit in correlate[cname][p][idx]:
					if hit in final[cn]:
						cnt += 1

				idx += 1
			print(['corr', cnt])
			if cnt >= params['Cutoffs']['Harmonics']['LeadCorrelates']:
				toadd.append(p)
				break

		return toadd

	@staticmethod
	def scoreit(params, cname, k, v):
		"""
		Based on processed parameters from an ECG peak, score it based on @params.
		Lead name @cname at time index @k and dictionary of processed parameters @v.
		"""
		p = []

		p.append(len(v['preRank']))
		p.append(len(v['sum/mean%max']))

		if sum(v['preRank'])/len(v['preRank']) > 0.45:
			p.append(2)
		elif sum(v['preRank'])/len(v['preRank']) > 0.25:
			p.append(1)
		else:
			p.append(0)

		if sum(v['sum/mean%max'])/len(v['sum/mean%max']) > 0.50:
			p.append(2)
		elif sum(v['sum/mean%max'])/len(v['sum/mean%max']) > 0.20:
			p.append(1)
		elif sum(v['sum/mean%max'])/len(v['sum/mean%max']) < 0.20:
			p.append(-2)
		elif sum(v['sum/mean%max'])/len(v['sum/mean%max']) < 0.10:
			p.append(-5)
		elif sum(v['sum/mean%max'])/len(v['sum/mean%max']) < 0.01:
			p.append(-50)
		else:
			p.append(0)

		if v['peakPercentile'] > 0.90:
			p.append(2)
		elif v['peakPercentile'] > 0.70:
			p.append(1)
		else:
			p.append(0)

		if v['varPercentile'] < params['Cutoffs']['varPercentile']['minimum'][0]:
			p.append(params['Cutoffs']['varPercentile']['minimum'][1])
		elif v['varPercentile'] > params['Cutoffs']['varPercentile']['desired'][0]:
			p.append(params['Cutoffs']['varPercentile']['desired'][1])
		else:
			p.append(0)

		return p

	def processecg_single(self, cname, dat, params, ignores, noises):
		"""
		Process a single lead named @cname based on series of samples in @dat and use parameters @params as needed.
		@rignores is a list of ranges in which samples should be ignored.
		"""

		# Steps to the algorithm:
		#  1) Skip this number
		#  2) Take d/dt
		#  3) Take Hilbert transform
		#  4) Calculate envelope function
		#  5) Perform averaging/smoothing filter
		#  6) Normalize to [0,1]
		#  7) Make more polarized
		#  8) Convolve a smoothing filter
		#  9) Take d/dt
		# 10) Find potential peaks
		# 11) Compare potential peaks to the raw data and find the local maximum in the raw data
		# 12) Calculate histogram for peaks
		# 13) Calculate the PDF of the peaks
		# 14) Update peak potentials with percentiles from (12)
		# 15) Calculate delta time between potentials

		len_dat = len(dat)
		min_dat = min(dat)
		print(['process', cname, len_dat, datetime.datetime.utcnow()])

		# 2)
		dat2 = [dat[i] - dat[i-1] for i in range(1,len_dat)] + [0]

		# 3)
		print(['A', cname, datetime.datetime.utcnow()])
		h = GenHilbert(params['HilbertLength'])
		f = list(ApplyFilter(dat2, h))

		# 4)
		print(['B', cname, datetime.datetime.utcnow()])
		f2 = [math.sqrt(f[i]*f[i] + dat2[i]*dat2[i]) for i in range(0,len(f))]
		# Delete, no longer needed
		del dat2

		# 5)
		print(['C', cname, datetime.datetime.utcnow()])
		f3 = ApplyFilter(f2, [1/params['Smoothing1']]*params['Smoothing1'])
		# Delete, no longer needed
		len_f2 = len(f2)
		del f2

		# 6)
		print(['D', cname, datetime.datetime.utcnow()])
		f3_min = min(f3)
		f3 = [_ - f3_min for _ in f3]
		f3_max = max(f3)
		f3 = [_ / f3_max for _ in f3]

		# 7)
		print(['E', cname, datetime.datetime.utcnow()])
		f4 = []
		for i in range(len(f3)):
			# Push towards zero or one to make more of a black/white selector
			if f3[i] < 0.5:
				f4.append( f3[i]*f3[i]*f3[i] )
			else:
				f4.append( 1-((1-f3[i])*(1-f3[i])*(1-f3[i])) )
		# Delete, no longer needed
		len_f3 = len(f3)
		del f3

		# 8)
		print(['F', cname, datetime.datetime.utcnow()])
		ones = [1] * params['Smoothing2']
		len_ones = len(ones)
		f5 = [0] * (len(f4) - len_ones)
		for i in range(len(f5)):
			f5[i] = sum(f4[i-len_ones:i]) / len_ones
		# Delete, no longer needed
		len_f4 = len(f4)
		del f4

		# 9)
		print(['G', cname, datetime.datetime.utcnow()])
		f6 = [f5[i+1] - f5[i] for i in range(len(f5)-1)]
		f6.append(0)
		# Delete, no longer needed
		len_f5 = len(f5)
		del f5
		len_f6 = len(f6)

		# 10A)
		print(['H', cname, datetime.datetime.utcnow()])
		potentials = {}
		f6_mean = sum(f6)/len(f6)

		widths = list(range(params['WindowWidthMax'],0,-5))
		# Calculate areas onces per wdith
		areas = {}
		for width in widths:
			areas[width] = math.fabs(f6_mean*(width-1)*2)

		max_width = max(widths)

		print(['I', cname, datetime.datetime.utcnow(), widths])
		for i in range(max_width, len_f6-max_width):
			# Extract the range once and pull portions of this for smaller widths
			pre_full = f6[i-max_width:i]
			post_full = f6[i:i+max_width]

			for width in widths:
				mean_area = areas[width]

				# Trapezoidal area calculation proportional to the sum of y-values (delta x is identical so don't bother)
				# Ends get summed once, but inner values get summed twice for adjacent trapezoids
				#trap = [1] + ([2]*(width-2)) + [1]

				# Get pre-window and post-window from the point f6[i] (zero crossing should have pre > 0 and post < 0)
				pre = pre_full[-width:]
				pre_area = 2*sum(pre) - pre[0] - pre[-1]
				if pre_area <= 0.02: continue

				post = post_full[:width]
				post_area = 2*sum(post) - post[0] - post[-1]
				if post_area >= -0.02: continue

				# Calculate ratio and invert so it's positive (ratio should be unity ideally)
				# then subtract 1 and so ratio is percent away from unity
				ratio = math.fabs( ((pre_area / post_area)*(-1)) - 1)
				if ratio > params['RatioCutoff']: continue

				# If reaches this point then all three are true
				# - If pre area is negative, absolutely don't count it
				# - If post area is positive, absolutely don't count it
				# - Ratio > X% of unity don't count it (parameter)

				# Window width that's wider is more supportive of being QRS
				# Areas that are larger are more supportive of being QRS
				# Ratio nearest to unity is more supportive of being QRS
				# sum/mean being large is more supportive of being QRS
				# sum/meam%max is percentile of highest sum/mean, higher the percentile is more supportive of being QRS
				potentials[i] = {'window': width, 'pre': pre_area, 'post': post_area, 'sum': pre_area + post_area, 'ratio': ratio, 'sum/mean': math.fabs(pre_area/mean_area)}

				potentials[i]['ignored'] = any([i in _ for _ in ignores])
				potentials[i]['noisy'] = any([i in _ for _ in noises])
				break

		# Delete, no longer needed
		del f6

		# 10B)
		# Calculate rank of pre-areas and normalize to [0,1]
		print(['J', cname, datetime.datetime.utcnow()])

		# Sort by pre area
		preareas = []
		for k,v in potentials.items():
			# Skip those ignored or noise
			if v['ignored'] or v['noisy']: continue

			preareas.append( (k, v['pre']) )
		preareas.sort(key=lambda _:_[1])
		preareas_len = len(preareas)

		# Get rank and normalize to [0,1]
		for idx,v in enumerate(preareas):
			k = v[0]
			potentials[k]['preRank'] = idx / preareas_len

		# 10C)
		# Calculate percentile of maximum sum/mean values
		print(['K', cname, datetime.datetime.utcnow()])
		max_summean = max([_['sum/mean'] for _ in potentials.values()])
		for k in potentials.keys():
			# Skip those ignored or noise
			if potentials[k]['ignored'] or potentials[k]['noisy']: continue

			potentials[k]['sum/mean%max'] = potentials[k]['sum/mean'] / max_summean

		# 11)
		print(['L', cname, datetime.datetime.utcnow()])
		winwidth = params['MaximumWindowWidth']
		maximums = {}
		for k in sorted(potentials.keys()):
			v = potentials[k]

			# Skip those ignored or noise
			if v['ignored'] or v['noisy']: continue

			#idx = k - params['Smoothing2'] - params['Smoothing1']
			idx = k - round(params['Smoothing2']/2)
			idx_min = int(idx-winwidth/2)
			idx_max = int(idx+winwidth/2)
			if idx_min < 0: idx_min = 0
			if idx_max > len_dat: idx_max = len_dat
			z = dat[idx_min:idx_max]
			avg = sum(z)/len(z)
			win = list(map(lambda _:math.fabs(_-avg), z))
			mx = max(win)
			idx = win.index(mx) + idx_min
			v['Maximum'] = {'idx': idx, 'value': mx}
			if idx not in maximums:
				maximums[idx] = 0
			maximums[idx] += 1

		# 12)
		print(['M', cname, datetime.datetime.utcnow()])
		maximums_histo = {}
		variances = {}
		for k,v in maximums.items():
			if v not in maximums_histo:
				maximums_histo[v] = 0
			maximums_histo[v] += 1

			idx = k
			idx_min = int(idx-winwidth/2)
			idx_max = int(idx+winwidth/2)
			if idx_min < 0: idx_min = 0
			if idx_max > len_dat: idx_max = len_dat
			win = list(map(lambda _:math.fabs(_), dat[idx_min:idx_max]))
			avg = sum(win)/len(win)
			var = sum([(_-avg)**2 for _ in win])/len(win)
			variances[idx] = var

		# 13)
		print(['N', cname, datetime.datetime.utcnow()])
		maximums_N = len(maximums.keys())
		maximums_p = {}
		maximums_pdf = {}
		running_sum = 0.0
		for k in sorted(maximums_histo):
			p = maximums_histo[k]/maximums_N
			maximums_p[k] = p
			running_sum += p
			maximums_pdf[k] = running_sum

		# 14)
		print(['O', cname, datetime.datetime.utcnow()])
		for k in potentials.keys():
			# Skip those ignored or noise
			if potentials[k]['ignored'] or potentials[k]['noisy']: continue

			# 'idx' is the index into @dat that is the local maximum for potential index of @k into @f6
			peak_N = maximums[ potentials[k]['Maximum']['idx'] ]
			# Thus, @peak_N is the number of times that @dat index showed up, so find it's percentile
			potentials[k]['Maximum']['Percentile'] = maximums_pdf[peak_N]
			# A wider potential set in @potentials that point to the same peak in @dat is supportive of being QRS

		print(['P', cname, datetime.datetime.utcnow()])
		peaks = {}
		for k in sorted(potentials.keys()):
			v = potentials[k]

			# Skip those ignored or noise
			if v['ignored'] or v['noisy']: continue

			idx = potentials[k]['Maximum']['idx']

			if idx not in peaks:
				var = variances[idx]
				peaks[idx] = {
					'potentials': [],
					'preRank': [],
					'sum/mean%max': [],
					'peakPercentile': v['Maximum']['Percentile'],
					'varPercentile': var,
				}

			peaks[idx]['potentials'].append(k)
			peaks[idx]['preRank'].append( v['preRank'] )
			peaks[idx]['sum/mean%max'].append( v['sum/mean%max'] )

		# 15)
		print(['Q', cname, datetime.datetime.utcnow()])
		last = 0
		for k in sorted(potentials.keys()):
			potentials[k]['deltaT'] = k - last
			last = k

		if False:
			i_start = 30000
			i_end = 31000
			r = range(i_start, i_end)

			pskeys = [_ for _ in peaks.keys() if _ in r]
			ps = {_:peaks[_] for _ in pskeys}

			fig,axs = plt.subplots(9)

			axs[0].plot(r, dat[i_start:i_end])
			# Plot as red circles ("ro") on top of the ECG data
			axs[0].plot(pskeys, [dat[_] for _ in pskeys], 'ro')

			axs[1].plot(r, dat2[i_start:i_end])
			axs[2].plot(r, f[i_start:i_end])
			axs[3].plot(r, f2[i_start:i_end])
			axs[4].plot(r, f3[i_start:i_end])
			axs[5].plot(r, f4[i_start:i_end])
			axs[6].plot(r, f5[i_start:i_end])
			axs[7].plot(r, f6[i_start:i_end])
			axs[8].plot(pskeys, [peaks[_]['varPercentile'] for _ in pskeys], 'ro')
			axs[8].xaxis.set_view_interval( *axs[7].xaxis.get_view_interval() )

			axs[0].set_ylabel("Data")
			axs[1].set_ylabel("d/dt")
			axs[2].set_ylabel("Hilbert Transformed")
			axs[3].set_ylabel("Envelope Function")
			axs[4].set_ylabel("Normalize to [0,1]")
			axs[5].set_ylabel("Noise Reduction")
			axs[6].set_ylabel("Apply Smoothing")
			axs[7].set_ylabel("d/dt")
			axs[8].set_ylabel("Peak Variance")

			fig.set_figwidth(20)
			fig.set_figheight(20)
			plt.savefig('snippet-%s.png' % cname, bbox_inches='tight')

		print(['R', cname, datetime.datetime.utcnow(), len(potentials), len(peaks)])
		return potentials, peaks

