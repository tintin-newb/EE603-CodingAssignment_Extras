import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import subprocess
from pickle import dump, dumps
from math import ceil, floor

'''
utility function to plot bar graphs
takes x-xoordinates and the height of the bars to be located
title needs to be provided
'''
def barplot_viz(x_int, x_float, c, interpolate_signal, title = 'Dummy'):

	plt.figure()
	plt.bar(x_int, c, width = 0.2, color = 'red')
	plt.bar(x_float, interpolate_signal, width = 0.2, color = 'green')
	labels = ['Sampled signal', 'Interpolated Signal']
	handles = [plt.Rectangle((0,0),1,1, color='red'), plt.Rectangle((0,0),1,1, color='green')]
	plt.legend(handles, labels)
	plt.ylabel('Intensity')
	plt.xlabel('Sampling/Interpolation Points')
	plt.title(title)
	plt.show()

'''
visualization of the sinc interpolation given some randomly spaced samples
total number of samples are 100
we take samples at an offset of 0.5 from integer positions as has been 
	described in the interpolation function in decoder

'''
def sinc_interpolation():
	nsamples = 100
	x_int = np.arange(0 ,100, 1)
	x_float = np.arange(0.5, 100, 1)
	#initialize sample signal that is 0 everywhere except at sampled instants
	c = np.zeros((100, ))			
	t = []
	#take sampling instances that are maximum of 5 samples apart
	#assign some random intensity value in range [0,255] in these instances to 'c' 
	tg = 5
	for i in range(20):
		t.append(np.random.randint(tg*i, tg*(i+1)))

	for t_idx in t:
		c[t_idx] = np.random.randint(0, 256)

	interpolate_signal = []
	for i in range(nsamples):
		n = i+0.5				#offset
		interpolate_signal.append(np.sum([c[idx]*np.sinc(idx-n) for idx in t]))

	interpolate_signal = np.clip(interpolate_signal, 0, 255)

	interpolate_signal1 = cv.GaussianBlur(interpolate_signal, (1, 5), sigmaX=1).reshape(interpolate_signal.shape)
	interpolate_signal2 = cv.GaussianBlur(interpolate_signal, (1, 5), sigmaX=6).reshape(interpolate_signal.shape)
	
	#plot sampled signals

	barplot_viz(x_int, x_float, c, interpolate_signal, "Interpolated Signal without Smoothing")
	barplot_viz(x_int, x_float, c, interpolate_signal1, "Smoothed Interpolated Signal with Sigma=1")
	barplot_viz(x_int, x_float, c, interpolate_signal2, "Smoothed Interpolated Signal with Sigma=6")

	return

#testing differences between adjacent frames using RMSE metric
def test_frame_diff():
	video_path = input('Enter video path: ')
	prev_frame = None
	first_frame = -1
	cap = cv.VideoCapture(video_path)
	score_lst = []
	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			if first_frame < 0:
				first_frame = 1
			else:
				row, col = frame.shape
				score = np.sum((frame-prev_frame)**2)/(row*col)
				score_lst.append(score)
			prev_frame = frame
		else:
			print("Capture Failed!")
			break

	cap.release()
	cv.destroyAllWindows()

	xpos = np.arange(1, len(score_lst)+1)
	plt.plot(xpos, score_lst)
	plt.ylabel("RMSE between k and k-1 frame")
	plt.title("RMSE Diff Score Plot")
	plt.show()


'''
pass the normalized and centered fourier transform of the signal and the corresponding frequencies
cutoff is the fraction of the max magnitude (1 as normalized)
bandwidth is the difference between low and high frequency for which the magnitude exceeds the cutoff
returns nyquist sampling rate based on the highest frequency and the bandwidth
'''
def maximal_bandwidth_calculation(ff, freq, cutoff = 0.05):

	n_samples = ff.shape[0]
	low, high = -1, -1
	for i in range(n_samples):
		if ff[i] >= cutoff:
			low = i 
			break

	for i in range(n_samples-1, -1, -1):
		if ff[i] >= cutoff:
			high = i
			break

	if low >= high:
		return -1, -1

	nyquist_rate = 2*freq[high]
	bandwidth = freq[high] - freq[low]

	return nyquist_rate, bandwidth

'''
traverse over pixels with stride and compute DFT
user has to input the shape of the image space separated as well as the kernel and stride
use maximal bandlimited signal as implemented in 'maximal_bandwidth_calculation()' to find the nyquist rate
the maximum Nyquist's raterate is noted
''' 
def signal_bandwidth_analysis():

	video_path = input("Enter input video path: ")
	max_nyq_rate = -1								#maximum nyquist's sampling frequency

	frame_store = []
	cap = cv.VideoCapture(video_path)
	nrow, ncol = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			frame_store.append(frame)
		else:
			print("Capture Failed or End of Stream?")
			break

	cap.release()
	frame_mat = np.stack(frame_store, axis = -1)		#stack the frames in a additional dimension at the end
	print(frame_mat.shape)

	for row in range(0, nrow):
		for col in range(0, ncol):
			signal = frame_mat[row][col][ : ]			#pixel signal across frames
			ff_mag, freq = dft_and_center(signal)
			nyquist_rate, __ = maximal_bandwidth_calculation(ff_mag, freq)
			if nyquist_rate == -1:						#signal is nearly constant
				print('Row:', row, 'Col:', col, 'Nearly constant!')
			max_nyq_rate = max(max_nyq_rate, nyquist_rate)

		print('Row',row,'Completed')


	print("Maximum Sampling Frequency: ", max_nyq_rate)
	print("Minimum Sampling Period: ", floor(1/max_nyq_rate))
	
	return

'''
compute the DFT of the signal and the correponding frequencies
the values of the fourier transform are normalized to have values in the range [0,1]
if paramter 'check' is set to True then the fourier magnitude and the frequencies are centered
returns the fourier magnitude and the corresponding frequencies
'''
def dft_and_center(signal, center=True):

	n_samples = signal.shape[0]
	#DFT magnitude
	ff = np.fft.fft(signal)
	ff_mag = np.abs(ff)
	ff_mag = ff_mag/np.amax(ff_mag)

	#frequencies
	freq = np.fft.fftfreq(n_samples)

	#center the DFT i.e. move DC (0) component to the center if set by 'check' parameter
	if center:
		ff_mag = np.fft.fftshift(ff_mag)
		freq = np.fft.fftshift(freq)

	return ff_mag, freq


'''
plot a line graph from the centered fourier transform magnitude
just for visualization purposes
'''
def visualize_dft(signal, position):

	ff_mag, freq = dft_and_center(signal)

	plt.plot(freq, ff_mag)
	plt.xlabel("Frequency")
	plt.ylabel("Magnitude")
	plt.title("DFT for pixel " + str(position[0]) + " " + str(position[1]))
	plt.show()

	return

'''
compute the fourier transform of a pixel (user input) and visualize the 
visualize the signal fourier transform
for demonstration purposes only
'''
def signal_analysis_viz():
	
	video_path = input("Enter input video path: ")
	stop_char = 'c'
	while stop_char != 'q':
		row, col = [int(tup) for tup in input("Pixel to analyze (row col): ").split(' ')]
		signal = []
		cap = cv.VideoCapture(video_path)
		while cap.isOpened():
			ret, frame = cap.read()
			if ret:
				frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
				signal.append(frame[row, col])
			else:
				"Capture Failed!"
				break

		cap.release()
		cv.destroyAllWindows()
		signal = np.array(signal)
		visualize_dft(signal, (row, col))
		stop_char = input('Press q to quit: ')


	return


'''
resize the image of the frames of the video
convert the frames to grayscale
NOTE: audio processing is not done 
'''
def resize_video_frames_grayscale():
	#modify the video stream

	video_path = input("Enter input video path: ")
	# f = int(input("Fraction to be reduced: "))
	cap = cv.VideoCapture(video_path)
	

	width = 500
	height = 300
	fps = int(cap.get(cv.CAP_PROP_FPS))
	vid_codec = cv.VideoWriter_fourcc('a', 'v', 'c', '1')
	print(width, height, fps, vid_codec)
	output = cv.VideoWriter('TestVid2.mp4', vid_codec, fps, (width, height), isColor=False)

	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			frame = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
			output.write(frame)
			cv.imshow('mod_frame', frame)
		else:
			"Capture failed!"
			break

		if cv.waitKey(1) == ord('q'):
			break


	cap.release()
	output.release()
	cv.destroyAllWindows()

	return


def main():

	idx = int(input("Process to run 1 to n: "))
	processes =  [resize_video_frames_grayscale, signal_analysis_viz, signal_bandwidth_analysis, test_frame_diff, sinc_interpolation]
	processes[idx-1]()

	print("Process " + str(idx) + " Completed")

if __name__ == "__main__":
	main()