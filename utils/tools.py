import re
import numpy as np

def get_age(raw_age):
    if '岁' in raw_age or '月' in raw_age or '日' in raw_age or '天' in raw_age:
        year = re.search(r'(\d*?)岁',raw_age)
        month = re.search(r'(\d*?)月',raw_age)
        day = re.search(r'(\d*?)日',raw_age)
        day2 = re.search(r'(\d*?)天',raw_age)

        ans = 0
        if year is None or year.group(1)=='': ans += 0
        else: ans += int(year.group(1))*365
        if month is None or month.group(1)=='': ans += 0
        else: ans += int(month.group(1))*30
        if day is None or day.group(1)=='': ans += 0
        else: ans += int(day.group(1))
        if day2 is None or day2.group(1)=='': ans += 0
        else: ans += int(day2.group(1))
        ans = ans // 365
    else:
        if 'Y' in raw_age:
            raw_age = raw_age.replace('Y','')
        try:
            ans = int(raw_age)
        except:
            ans = -1
    if ans < 0:
        return ''
    elif ans >= 0 and ans < 1:
        return '婴儿'
    elif ans >= 1 and ans <= 6:
        return '童年'
    elif ans >=7 and ans <= 18:
        return '少年'
    elif ans >= 19 and ans <= 30:
        return '青年'
    elif ans >= 31 and ans <= 40:
        return '壮年' 
    elif ans >= 41 and ans <= 55:
        return '中年'
    else:
        return '老年'

def find_threshold_micro(dev_yhat_raw, dev_y):
    dev_yhat_raw_1 = dev_yhat_raw.reshape(-1)
    dev_y_1 = dev_y.reshape(-1)
    sort_arg = np.argsort(dev_yhat_raw_1)
    sort_label = np.take_along_axis(dev_y_1, sort_arg, axis=0)
    label_count = np.sum(sort_label)
    correct = label_count - np.cumsum(sort_label)
    predict = dev_y_1.shape[0] + 1 - np.cumsum(np.ones_like(sort_label))
    f1 = 2 * correct / (predict + label_count)
    sort_yhat_raw = np.take_along_axis(dev_yhat_raw_1, sort_arg, axis=0)
    f1_argmax = np.argmax(f1)
    threshold = sort_yhat_raw[f1_argmax]
    # print(threshold)
    return threshold