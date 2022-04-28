from threading import Thread

class MyThread(Thread):
    '''Multiple threads wrapper'''
    def __init__(self, func, args):
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result


def wave_analysis(histogram:list, threshold:float, low_granularity:int, high_granularity:int) -> list:
    '''wave analysis
    Params:
        histogram: a wave
        threshold: the threshold
        low_granularity: the low granularity of a wave peak, if lower than threshold in peak smaller than low_granularity, ignored
        high_granularity: the high granularity of a wave peak, only wave length higher than high_granularity will be accepted
    ----------
    Return:
        wave_peaks: the list of peaks, each one with up_point and down_point
    ----------
    Note:
        the constraint is somehow strict
    '''
    up_point = -1
    in_peak = False
    if histogram[0] > threshold:
        up_point = 0
        in_peak = True
    
    wave_peaks = []
    for i, x in enumerate(histogram):
        if not in_peak:
            if x >= threshold:
                # start a new peak
                up_point = i
                in_peak = True
            else:
                # stay in the background
                continue
        else:
            if x < threshold:
                if i - up_point <= low_granularity:
                    # ignore the noise
                    continue
                else:
                    if i - up_point <= high_granularity:
                        # refuse this peak
                        in_peak = False
                        continue
                    else:
                        # accept this peak
                        down_point = i
                        wave_peaks.append((up_point, i))
                        in_peak = False
            else:
                # stay in the peak
                continue
    # deal with the last peak
    if in_peak and up_point != -1 and len(histogram)-1-up_point > high_granularity:
        down_point = len(histogram)-1
        wave_peaks.append((up_point, down_point))
        in_peak = False
    
    return wave_peaks


def digit2char(charactor:str) -> str:
    '''convert a digit charactor to possible English charactor
    Params:
        charactor: a digit or English charactor
    ----------
    Return:
        charactor: English charactor
    '''
    if charactor == '0':
        charactor = 'O'
    elif charactor == '1':
        charactor = 'I'
    elif charactor == '2':
        charactor = 'Z'
    elif charactor == '3':
        charactor = 'E'
    elif charactor == '4':
        charactor = 'A'
    elif charactor == '5':
        charactor = 'S'
    elif charactor == '6':
        charactor = 'G'
    elif charactor == '7':
        charactor = 'T'
    elif charactor == '8':
        charactor = 'B'
    elif charactor == '9':
        charactor = 'P'
    else:
        charactor = charactor
    
    return charactor

def char2digit(charactor:str) -> str:
    '''convert to a English charactor to possible digit charactor
    Params:
        charactor: a digit or English charactor
    ----------
    Return:
        charactor: digit or English charactor
    '''
    if charactor == 'O':
        charactor = '0'
    elif charactor == 'I':
        charactor = '1'
    elif charactor == 'Z':
        charactor = '2'
    elif charactor == 'E':
        charactor = '3'
    elif charactor == 'A':
        charactor = '4'
    elif charactor == 'S':
        charactor = '5'
    elif charactor == 'G':
        charactor = '6'
    elif charactor == 'T':
        charactor = '7'
    elif charactor == 'B':
        charactor = '8'
    elif charactor == 'P':
        charactor = '9'
    else:
        charactor = charactor
    
    return charactor