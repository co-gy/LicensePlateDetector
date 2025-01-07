def return_ccpd_info(filename):
    """
    获取CCPD标签的信息
    :filename: 标签文件名（不带后缀）
    """
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫",
                 "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W', 'X', 'Y', 'Z', 'O']
    ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
    _, _, box, points, plate, brightness, blurriness = filename.split('-')
    # 车牌号
    list_plate = plate.split('_')
    province = provinces[int(list_plate[0])]    # 省份
    alphabet = alphabets[int(list_plate[1])]    # 城市
    list_plate = (province + alphabet + ads[int(list_plate[2])] + ads[int(list_plate[3])] +
                  ads[int(list_plate[4])] + ads[int(list_plate[5])] + ads[int(list_plate[6])])
    # print("车牌号:",list_plate)
    return list_plate