import io
import math
import os
import pickle

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
			'Smoothing1': 10,
			'Smoothing2': 20,
			'WindowWidthMax': 50,
			'MaximumWindowWidth': 40,
			'RatioCutoff': 0.02,
			'Cutoffs': {
				'varPercentile': {
					'minimum': (1000.0, -100),
					'desired': (5000.0, +1),
				},
				'Points': 4,
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
		for fidx,chans,samps in self.WIFF.recording[1].GetAllFrames():
			if dat is None:
				# Avoid doing this every iteration since it doesn't change
				i = len(chans)
				dat = [[]] * i
				potentials = {_:[] for _ in chans}
				peaks = {_:[] for _ in chans}

			for n in range(i):
				dat[n].append(samps[n])

		# Data loaded, now process
		for cidx,cname in enumerate(chans):
			potentials[cname], peaks[cname] = processecg_single(cname, dat[cidx], self.Params, rignores, rnoise)

		return (potentials, peaks)

	def GetCorrelate(self, chans, potentials, peaks, params):
		return correlateleads(chans, potentials, peaks, params)

	def GetKeepKeys(self, chans, peaks):
		keep_keys = {k:[] for k in peaks.keys()}
		points = {k:{} for k in peaks.keys()}

		# 4)
		# NOTE: This is currently just empirically determined, better if some sort of non-linear model (ie, "AI") were used instead
		for cname in peaks.keys():
			for k in sorted(peaks[cname].keys()):
				v = peaks[cname][k]

				p = scoreit(params, cname, k, v)
				points[cname][k] = p

				if sum(p) >= params['Cutoffs']['Points']:
					print([cname, k, k/2000, sum(p), p, 'keep'])
					keep_keys[cname].append(k)
				else:
					print([cname, k, k/2000, sum(p), p, 'pass'])

		return (points, keep_keys)

	def GetRemoveKeys(self, chans, correlate, points, keep):
		remove = []
		for cname,v in keep_keys.items():
			for k in v:
				# Get correlated indices and current points for this peak
				corr = correlate[cname][k]
				corr_len = [len(_) > 0 for _ in corr].count(True)
				p = points[cname][k]

				# Iterate thrugh other leads and check how
				idx = 0
				cnt = 0
				for n in header[1:]:
					if n == cname: continue

					# Check that any of the peaks correlated for each lead scored enough points to be kept
					if any([_ for _ in corr[idx] if _ in keep_keys[n]]):
						if cname == 'Lead I' and k == 112925:
							print(['Correlate lead yes', cname, k, n])
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
		return

	def CalculateRR(self, chans, keep, remove, user, intervals, noise, params):
		"""
		Calculate RR based on the intervals provided.
		@intervals is a dictionary mapping named regions to a list of (start,stop) frames that are in that named region.
		@noise is a list of (start,stop) regions that are ignored from analysis as user indicates it is noise.
		"""

		# 6A)
		RR = []
		for cname in keep.keys():
			RR += [keep[cname][i] - keep[cname][i-1] for i in range(1,len(keep[cname]))]
		min_RR = min(RR)

		# 6B)
		histo = {}
		for k in RR:
			if k not in histo: histo[k] = 0
			histo[k] += 1

		# 6C)
		# The fundamental frequency of a dataset with harmonics necessarily is less than mean of the dataset
		# (If the dataset contained only a single harmonic then the mean is the fundamental frequency)
		# So the fundamental frequency is limited to min(histo.keys()) to avg
		avg = sum([k*v for k,v in histo.items()])/sum(histo.values())

		# 6D)
		# Get the average of the first harmonic, this average is likely pretty close
		z = [_ for _ in RR if _ < int(avg)+1]
		del RR
		f0 = sum(z)/len(z)

		# 6E)
		# Now go back through ECG data and find gaps larger than f0, particularly if an integer multiple suggesting missed complexes
		too_soon = {k:[] for k in header[1:]}
		for cname in keep.keys():
			last_k = None
			for k in sorted(keep[cname]):
				# Skip first
				if last_k is None:
					last_k = k
					continue

				h = (k-last_k)/f0
				mult = round(h)
				print([cname, k, last_k, k-last_k, h, mult])

				if last_k in [_[0] for _ in too_soon[cname]]:
					print("Don't penalize next peak after Too Soon point")
					last_k = k
					continue

				if h < 0.85:
					# Weird point, probably PAC/PVC or P wave
					print("Too soon")
					too_soon[cname].append( (k, last_k, h, mult) )
					pass
				elif h < 1.15:
					# First harmonic, skip it
					pass
				else:
					print("Harmonic")
					print([last_k, k, mult, list(harmonicrange(last_k, k, mult))])
					for idx in harmonicrange(last_k, k, mult):
						findmissingharmonicpeak(peaks, keep, correlate, cname, last_k, k, idx, params, chans)

				# Update index for next loop
				last_k = k

		# 6F)
		# TODO: for now just remove them
		print(too_soon)
		for cname in too_soon.keys():
			for k, last_k, h, mult in too_soon[cname]:
				keep_keys[cname].remove(k)

	def OldCaclulateRR(self):

		# Things to do:
		#  1) Read in data
		#  2) Correlate peaks against peaks in other leads within params['LeadCorrelateWindow'] window
		#  3) Index list of keys (indices) to keep, which is keyed off of the lead name
		#     Also keep a similar list of point scores for each criteria for each peak
		#  4) Iterate through all peaks and filter some out based on paramters
		#  5) Remove peaks based on lake of correlation with other leads
		#  6) Fill in gaps using RR intervals and concept of harmonics
		#     6A) Calculate RR intervals
		#     6B) Calculate histogram of RR intervals
		#     6C) Average of histogram should be greater (or equal to) fundamental harmonic
		#     6D) Truncate histogram to (6C) average is assumed to be close to fundamental RR interval
		#     6E) Find integer gaps and fill them in if able
		#     6F) If a peak is non-harmonic then assume it's a P wave or PVC or something and just ignore it
		# 7) Export to PNG
		# 8) Calculate R-R intervals

		# 1)
		fname_dat = self.Filename + '.pypickle'
		if self.CacheExists():
			z = self.CacheRead()
			header = z['header']
			potentials = z['potentials']
			peaks = z['peaks']
			header, dat = loadecg(fname)

		else:
			# Process the ECG data to get peak information
			# peaks is indexed by column name
			header, dat, potentials, peaks = processecg(fname, params)

			self.CacheWrite({'header': header, 'potentials': potentials, 'peaks': peaks})

		# 2)
		correlate = correlateleads(header, potentials, peaks, params)

		# Print out all peaks for debugging
		if False:
			for cname in peaks.keys():
				for k in sorted(peaks[cname]):
					print([cname, k, peaks[cname][k]])

		# 3)
		keep_keys = {k:[] for k in peaks.keys()}
		points = {k:{} for k in peaks.keys()}

		# 4)
		# NOTE: This is currently just empirically determined, better if some sort of non-linear model (ie, "AI") were used instead
		for cname in peaks.keys():
			for k in sorted(peaks[cname].keys()):
				v = peaks[cname][k]

				p = scoreit(params, cname, k, v)
				points[cname][k] = p

				if sum(p) >= params['Cutoffs']['Points']:
					print([cname, k, k/2000, sum(p), p, 'keep'])
					keep_keys[cname].append(k)
				else:
					print([cname, k, k/2000, sum(p), p, 'pass'])

		# 5)
		remove = []
		for cname,v in keep_keys.items():
			for k in v:
				# Get correlated indices and current points for this peak
				corr = correlate[cname][k]
				corr_len = [len(_) > 0 for _ in corr].count(True)
				p = points[cname][k]

				# Iterate thrugh other leads and check how
				idx = 0
				cnt = 0
				for n in header[1:]:
					if n == cname: continue

					# Check that any of the peaks correlated for each lead scored enough points to be kept
					if any([_ for _ in corr[idx] if _ in keep_keys[n]]):
						if cname == 'Lead I' and k == 112925:
							print(['Correlate lead yes', cname, k, n])
						cnt += 1

					idx += 1

				# This peak only matches 0-2 other leads, probably not enough to be a QRS so get rid of it
				if cnt < 3:
					remove.append( (cname,k) )

				if cname == 'Lead I' and k == 112925:
					print(['corr', corr, corr_len, p, cnt])
					print([(cname, k), n, cnt, 'remove'])

		# Remove them
		for cname,k in remove:
			keep_keys[cname].remove(k)

		# 6A)
		RR = []
		for cname in keep_keys.keys():
			RR += [keep_keys[cname][i] - keep_keys[cname][i-1] for i in range(1,len(keep_keys[cname]))]
		min_RR = min(RR)

		# 6B)
		histo = {}
		for k in RR:
			if k not in histo: histo[k] = 0
			histo[k] += 1

		# 6C)
		# The fundamental frequency of a dataset with harmonics necessarily is less than mean of the dataset
		# (If the dataset contained only a single harmonic then the mean is the fundamental frequency)
		# So the fundamental frequency is limited to min(histo.keys()) to avg
		avg = sum([k*v for k,v in histo.items()])/sum(histo.values())

		# 6D)
		# Get the average of the first harmonic, this average is likely pretty close
		z = [_ for _ in RR if _ < int(avg)+1]
		del RR
		f0 = sum(z)/len(z)

		# 6E)
		# Now go back through ECG data and find gaps larger than f0, particularly if an integer multiple suggesting missed complexes
		too_soon = {k:[] for k in header[1:]}
		for cname in keep_keys.keys():
			last_k = None
			for k in sorted(keep_keys[cname]):
				# Skip first
				if last_k is None:
					last_k = k
					continue

				h = (k-last_k)/f0
				mult = round(h)
				print([cname, k, last_k, k-last_k, h, mult])

				if last_k in [_[0] for _ in too_soon[cname]]:
					print("Don't penalize next peak after Too Soon point")
					last_k = k
					continue

				if h < 0.85:
					# Weird point, probably PAC/PVC or P wave
					print("Too soon")
					too_soon[cname].append( (k, last_k, h, mult) )
					pass
				elif h < 1.15:
					# First harmonic, skip it
					pass
				else:
					print("Harmonic")
					print([last_k, k, mult, list(harmonicrange(last_k, k, mult))])
					for idx in harmonicrange(last_k, k, mult):
						findmissingharmonicpeak(peaks, keep_keys, correlate, cname, last_k, k, idx, params, header)

				# Update index for next loop
				last_k = k

		# 6F)
		# TODO: for now just remove them
		print(too_soon)
		for cname in too_soon.keys():
			for k, last_k, h, mult in too_soon[cname]:
				keep_keys[cname].remove(k)

		if False:
			print("Kept")
			print("index\tTime\tpreRank.N\tpreRank.min\tpreRank.max\tpreRank.mean\tsum/mean%max.N\tsum/mean%max.min\tsum/mean%max.max\tsum/mean%max.mean\tpeakPercentile\tvarPercentile")
			for k in keep_keys:
				_ = peaks[k]
				print(k, end='\t')
				print(k/2000, end='\t')
				print(len(_['preRank']), end='\t')
				print(min(_['preRank']), end='\t')
				print(max(_['preRank']), end='\t')
				print(sum(_['preRank'])/len(_['preRank']), end='\t')
				print(len(_['sum/mean%max']), end='\t')
				print(min(_['sum/mean%max']), end='\t')
				print(max(_['sum/mean%max']), end='\t')
				print(sum(_['sum/mean%max'])/len(_['sum/mean%max']), end='\t')
				print(_['peakPercentile'], end='\t')
				print(_['varPercentile'])

			print()
			print("Rejected")
			print("index\tTime\t\tpreRank.N\tpreRank.min\tpreRank.max\tpreRank.mean\tsum/mean%max.N\tsum/mean%max.min\tsum/mean%max.max\tsum/mean%max.mean\tpeakPercentile\tvarPercentile")
			for k in [_ for _ in peaks.keys() if _ not in keep_keys]:
				_ = peaks[k]
				print(k, end='\t')
				print(k/2000, end='\t')
				print(len(_['preRank']), end='\t')
				print(min(_['preRank']), end='\t')
				print(max(_['preRank']), end='\t')
				print(sum(_['preRank'])/len(_['preRank']), end='\t')
				print(len(_['sum/mean%max']), end='\t')
				print(min(_['sum/mean%max']), end='\t')
				print(max(_['sum/mean%max']), end='\t')
				print(sum(_['sum/mean%max'])/len(_['sum/mean%max']), end='\t')
				print(_['peakPercentile'], end='\t')
				print(_['varPercentile'])
			print()

		# 7)
		ret = ecgtopng(header, dat, keep_keys, self.WritePNG, width=params['PNG']['width'], speed=params['PNG']['speed'])

		# 8)
		cname = list(keep_keys.keys())[0]
		RR = [keep_keys[cname][i] - keep_keys[cname][i-1] for i in range(1,len(keep_keys[cname]))]
		for r in RR:
			print(r)

def findmissingharmonicpeak(peaks, keep_keys, correlate, cname, last_k, k, idx, params, header):
	"""
	Try to find a peak for lead @cname around index @idx, which was found using harmonic analysis between indices @last_k and @k.
	"""

	winwidth = params['LeadCorrelateWindow']
	r = range(idx - winwidth, idx + winwidth)

	#print(['find peak', cname, idx, r])
	possibles = [_ for _ in peaks[cname].keys() if _ in r]
	#print(['possibles', possibles])

	for p in possibles:
		score = scoreit(params, cname,p, peaks[cname][p])
		print([p, peaks[cname][p], score])

		cnt = 0
		idx = 0
		for cn in header[1:]:
			if cn == cname: continue

			for hit in correlate[cname][p][idx]:
				if hit in keep_keys[cn]:
					cnt += 1

			idx += 1
		#print(['corr', cnt])
		if cnt >= 3:
			keep_keys[cname].append(p)
			keep_keys[cname].sort()
			break

def correlateleads(header, potentials, peaks, params):
	"""
	Correlated peaks across leads (they don't always happen at the same time index.
	"""

	keys = {}
	correlate = {}
	for cname in header[1:]:
		keys[cname] = sorted(peaks[cname].keys())

	winwidth = params['LeadCorrelateWindow']
	for mainkey in header[1:]:
		correlate[mainkey] = {}

		for k in keys[mainkey]:
			r = range(k-winwidth, k+winwidth+1)
			correlate[mainkey][k] = []

			for cname in header[1:]:
				if cname == mainkey: continue

				correlate[mainkey][k].append( [_ for _ in peaks[cname] if _ in r] )

	return correlate

def scoreit(params, cname, k, v):
	"""
	Based on processed parameters from an ECG peak, score it based on @params.
	Lead name @cname at time index @k and dictionary of processed parameters @v.
	"""
	p = []

	if len(v['preRank']) <= 3:
		p.append(-100)
	elif len(v['preRank']) == 4:
		p.append(0)
	elif len(v['preRank']) == 5:
		p.append(1)
	elif len(v['preRank']) >= 6 and len(v['preRank']) <= 10:
		p.append(1)
	elif len(v['preRank']) > 10:
		p.append(20)
	else:
		p.append(0)

	if len(v['sum/mean%max']) <= 3:
		p.append(-100)
	elif len(v['sum/mean%max']) == 4:
		p.append(0)
	elif len(v['sum/mean%max']) == 5:
		p.append(1)
	elif len(v['sum/mean%max']) >= 6 and len(v['sum/mean%max']) <= 10:
		p.append(1)
	elif len(v['sum/mean%max']) > 10:
		p.append(20)
	else:
		p.append(0)

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

def ecgtopng(header, dat, peaks, writefunc, width=20, speed=200, outdir=None):
	"""
	Convert an ECG to a series of PNG images for slices of the ECG data into @width inches (matlabplot saves in inches) and at @speed mm/sec (sorry, EKG's are printed in mm/sec and I'm not going to make it in inches).
	The header row @header is used to provide the lead names ([0] should be "Time").
	The data @dat is a list of data points in time with each data point as the [time, lead I, lead II, ....] at that time point.
	The RR data @peaks to plot dots on the ECG data that is a list of time indices of @dat.
	"""

	if len(dat[0]) != 7:
		raise ValueError("ECG data has more than 6 columns (%d), don't know what to do with that" % len(dat[0]))

	# 6 leads so 6 vertical sub plots
	fig,axs = plt.subplots(6)

	# Get the start and end times of the data
	tstart = dat[0][0]
	tend = dat[-1][0]

	# Total time delta across all leads
	delta = tend - tstart

	# Sampling rate in Hz would be total time divided by samples (Hz)
	f = len(dat) / delta

	# Calculate samples per page with 25.4 mm = 1 inch
	samps = (width * 25.4) / speed * f

	# Truncate to whole sample to step
	step = int(samps)

	ret = {
		'start': tstart,
		'end': tend,
		'delta': delta,
		'samplesperpage': step,
		'samplingrate': f,
	}

	# Iterate over @width inches of ECG per figure
	for idx in range(0, len(dat), step):
		s = slice(idx, idx+step)
		r = range(idx, idx+step)

		# Slice of time data to plot
		d = dat[s]

		# Iterate over each lead in the time slice
		for i in range(1,len(dat[0])):
			cname = header[i]

			# Clear the subfigure
			axs[i-1].cla()
			plt.xlabel("%s (sec)" % header[0])

			# Apply lead label to the y axis
			axs[i-1].set_ylabel(cname)

			# pull out the column of data to plot
			subd = [_[i] for _ in d]
			axs[i-1].plot(list(map(lambda _:_/f, range(idx, idx+len(d)))), subd)
			if i < len(dat[0])-1:
				axs[i-1].get_xaxis().set_visible(False)
			else:
				axs[i-1].get_xaxis().set_visible(True)

			# Filter out the peaks data for the time slice
			match_points = [_ for _ in peaks[cname] if _ in r]

			# Plot as red circles ("ro") on top of the ECG data
			axs[i-1].plot([_/f for _ in match_points], [subd[_-idx] for _ in match_points], 'ro')

			# Every quarter second to two decimal points, and 4 minor tick marks per major
			axs[i-1].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
			axs[i-1].xaxis.set_major_formatter('{x:.2f}')
			axs[i-1].xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.025))

		# Format figure size
		fig.set_figwidth(width)
		fig.set_figheight(12)

		with io.BytesIO() as f:
			# Save figure
			plt.savefig(f, format='png', bbox_inches='tight')

			f.seek(0)
			writefunc(idx/step, f)

	return ret

def loadecg(fname):
	"""
	Loads the raw data from @fname and returns the header row and data rows.
	"""

	dat = []
	row = 2
	last_time = -1
	with open(fname, 'r') as f:
		header = f.readline()
		header = [_.strip() for _ in header.split('\t')]
		# Prefix all leads with "Lead" to be consistent
		header2 = [header[0]] # Skip [0] == Time
		for v in header[1:]:
			if v.startswith('Lead'): header2.append(v)
			else: header2.append("Lead " + v)
		header = header2
		del header2

		ln_header = len(header)
		while True:
			line = f.readline()
			if not len(line):
				break

			z = [float(_.strip()) for _ in line.split('\t')]
			if len(z) != ln_header:
				print(z)
				raise ValueError("Row %d should have %d columns but has %d" % (row, ln_header, len(z)))
			if z[0] < last_time:
				raise ValueError("Row %d time is %f but prior row had time %f and should be bigger" % (row, z[0], last_time))
			row += 1
			last_time = z[0]
			dat.append(z)

	return header, dat

def processecg(fname, params):
	"""
	Process the ECG file at @fname with parsing parameters @params.
	Returns the tuple (headers column names, raw data, potential peaks, and filtered peaks).
	"""

	# Get header row and raw data
	header, dat = loadecg(fname)

	# Process each lead to get potentials and filtered peaks
	potentials = {}
	peaks = {}
	for colnum in range(1,len(dat[0])):
		# Column name
		cname = header[colnum]
		print("Processing %s (%d of %d)" % (cname, colnum, len(dat[0])-1))

		# Get lead data
		col = []
		for row in dat:
			col.append(row[colnum])

		potentials[cname],peaks[cname] = processecg_single(cname, col, params)

	return header, dat, potentials, peaks

def processecg_single(cname, dat, params, ignores, noises):
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

	# 2)
	dat2 = [dat[i] - dat[i-1] for i in range(1,len(dat))] + [0]

	# 3)
	h = GenHilbert(params['HilbertLength'])
	f = list(ApplyFilter(dat2, h))

	# 4)
	f2 = [math.sqrt(f[i]*f[i] + dat2[i]*dat2[i]) for i in range(0,len(f))]

	# 5)
	f3 = ApplyFilter(f2, [1/params['Smoothing1']]*params['Smoothing1'])

	# 6)
	f3_min = min(f3)
	f3 = [_ - f3_min for _ in f3]
	f3_max = max(f3)
	f3 = [_ / f3_max for _ in f3]

	# 7)
	f4 = []
	for i in range(len(f3)):
		# Push towards zero or one to make more of a black/white selector
		if f3[i] < 0.5:
			f4.append( f3[i]*f3[i]*math.fabs(dat[i]) )
		else:
			f4.append( (1-f3[i])*(1-f3[i])*math.fabs(dat[i]) )

	# 8)
	ones = [1] * params['Smoothing2']
	f5 = [0] * (len(f4) - len(ones))
	for i in range(len(f5)):
		for k in range(len(ones)):
			f5[i] += f4[i-k-len(ones)] * ones[k]
		f5[i] /= len(ones)

	# 9)
	f6 = [f5[i+1] - f5[i] for i in range(len(f5)-1)]
	f6.append(0)

	# 10A)
	potentials = {}
	f6_mean = sum(f6)/len(f6)

	maxes = []
	widths = list(range(params['WindowWidthMax'],0,-5))
	for width in widths:
		# Area based on mean f6 values so this would be the trapezoidal "area" for an average window over f6
		mean_area = f6_mean*(width-1)*2

		for i in range(width, len(f6)-width):
			# If already recorded for a wider window, skip it
			if i in potentials: continue

			# Get pre-window and post-window from the point f6[i] (zero crossing should have pre > 0 and post < 0)
			pre = f6[i-width:i]
			post = f6[i:i+width]

			# Trapezoidal area calculation proportional to the sum of y-values (delta x is identical so don't bother)
			# Ends get summed once, but inner values get summed twice for adjacent trapezoids
			trap = [1] + ([2]*(width-2)) + [1]

			# Calculate areas
			pre_area = sum([pre[_]*trap[_] for _ in range(width)])
			post_area = sum([post[_]*trap[_] for _ in range(width)])
			# Calculate ratio and invert so it's positive (ratio should be unity ideally)
			# then subtract 1 and so ratio is percent away from unity
			ratio = math.fabs(pre_area / post_area * (-1) - 1)

			# If pre area is negative, absolutely don't count it
			# If post area is positive, absolutely don't count it
			# Ratio > X% of unity don't count it (parameter)
			if pre_area > 0 and post_area < 0 and ratio < params['RatioCutoff']:
				maxes.append(i)

				# Window width that's wider is more supportive of being QRS
				# Areas that are larger are more supportive of being QRS
				# Ratio nearest to unity is more supportive of being QRS
				# sum/mean being large is more supportive of being QRS
				# sum/meam%max is percentile of highest sum/mean, higher the percentile is more supportive of being QRS
				potentials[i] = {'window': width, 'pre': pre_area, 'post': post_area, 'sum': pre_area + post_area, 'ratio': pre_area/post_area, 'sum/mean': pre_area/mean_area}

	# 10B)
	# Calculate rank of pre-areas and normalize to [0,1]
	preareas = [_['pre'] for _ in potentials.values()]
	preareas.sort()
	for k in potentials.keys():
		potentials[k]['preRank'] = preareas.index(potentials[k]['pre']) / len(preareas)

	# 10C)
	# Calculate percentile of maximum sum/mean values
	max_summean = max([_['sum/mean'] for _ in potentials.values()])
	for k in potentials.keys():
		potentials[k]['sum/mean%max'] = potentials[k]['sum/mean'] / max_summean

	# 11)
	winwidth = params['MaximumWindowWidth']
	maximums = {}
	for k in sorted(potentials.keys()):
		v = potentials[k]
		idx = k - params['Smoothing2'] - params['Smoothing1']
		idx_min = int(idx-winwidth/2)
		idx_max = int(idx+winwidth/2)
		if idx_min < 0: idx_min = 0
		if idx_max > len(dat): idx_max = len(dat)
		win = list(map(lambda _:math.fabs(_), dat[idx_min:idx_max]))
		mx = max(win)
		idx = win.index(mx) + idx_min
		v['Maximum'] = {'idx': idx, 'value': mx}
		if idx not in maximums:
			maximums[idx] = 0
		maximums[idx] += 1

	# 12)
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
		if idx_max > len(dat): idx_max = len(dat)
		win = list(map(lambda _:math.fabs(_), dat[idx_min:idx_max]))
		avg = sum(win)/len(win)
		var = sum([(_-avg)**2 for _ in win])/len(win)
		variances[idx] = var

	# 13)
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
	for k in potentials.keys():
		# 'idx' is the index into @dat that is the local maximum for potential index of @k into @f6
		peak_N = maximums[ potentials[k]['Maximum']['idx'] ]
		# Thus, @peak_N is the number of times that @dat index showed up, so find it's percentile
		potentials[k]['Maximum']['Percentile'] = maximums_pdf[peak_N]
		# A wider potential set in @potentials that point to the same peak in @dat is supportive of being QRS

	peaks = {}
	for k in sorted(potentials.keys()):
		v = potentials[k]
		idx = potentials[k]['Maximum']['idx']

		potentials[k]['ignored'] = False
		potentials[k]['noisy'] = False

		# Instructed to ignore it
		if any([idx in _ for _ in ignores]):
			potentials[k]['ignored'] = True
			continue
		if any([idx in _ for _ in noises]):
			potentials[k]['noisy'] = True
			continue

		# Not ignored or in band of noise

		if idx not in peaks:
			var = variances[idx]
			peaks[idx] = {
				'preRank': [],
				'sum/mean%max': [],
				'peakPercentile': v['Maximum']['Percentile'],
				'varPercentile': var,
			}

		peaks[idx]['preRank'].append( v['preRank'] )
		peaks[idx]['sum/mean%max'].append( v['sum/mean%max'] )

	# 15)
	last = 0
	for k in sorted(potentials.keys()):
		potentials[k]['deltaT'] = k - last
		last = k

	if False:
		i_start = 4060
		i_end = 4060*2
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
		plt.savefig('snippet.png', bbox_inches='tight')

	return potentials, peaks

