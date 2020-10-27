pairwise_clusters_dict = {
        'CC': [[1289.05262549],
            [2192.48755187],
            [3682.22057747],
            [2650.04420745],
            [2533.26445159],
            [1522.5746184 ],
            [2396.57013118]],
        'HC': [[2185.97914941],
            [1092.40201697]],
        'HH': [[1773.73995018],
            [3798.99906388],
            [2514.29220833],
            [4293.69251428],
            [3079.0113923 ]],
        'CO': [[3699.49351464],
            [3464.78185583],
            [3185.35056439],
            [1204.60388888],
            [1412.18916842],
            [2442.03840065],
            [2156.3794724 ],
            [2922.27895993]],
        'HO': [[4067.24546648],
            [ 963.77124601],
            [2604.58059049],
            [2712.59445186],
            [2071.06125203],
            [3337.86799692]],
        'OO': [[2265.41722277],
            [4277.6124902 ],
            [3274.40477687],
            [4614.96711202],
            [4132.37122417],
            [2816.41227842],
            [3621.94426634],
            [2697.48667578],
            [2971.2834873 ],
            [3470.1178997 ]],
        'CN': [[3555.90521923],
            [1435.93060675],
            [2218.25826741],
            [2447.26416228],
            [1250.9734164 ],
            [2571.95610336]],
        'HN': [[4352.08088721],
            [2769.56091578],
            [3174.41187699],
            [3410.61063396],
            [2131.66385048],
            [4107.93280248],
            [1012.265085  ]],
        'NO': [[2306.4621979 ],
                [4746.22912475],
                [4067.32391714],
                [4251.82940622],
                [3597.33419101],
                [3103.93688717],
                [3402.84121748],
                [4536.85025333],
                [1399.74203205],
                [2809.52351575]],
        'NN': [[3615.99803683],
                [2797.18826483],
                [1329.58966707],
                [4117.17470782],
                [4709.74987654],
                [3458.38565974],
                [2376.86833874],
                [2231.16287656]],
        'CF': [[3566.37053392],
                [1332.17227662],
                [2890.91825613],
                [2354.73729267],
                [4043.76856436]],
        'HF': [[4508.7883959 ],
                [2605.79121278],
                [4903.89479638],
                [4029.24096386],
                [3691.18958611],
                [4689.15841584],
                [3479.53125   ],
                [2374.90566038],
                [4316.29291045],
                [3243.84184184],
                [3879.99049881],
                [2831.28187919]],
        'NF': [[3562.21524664],
                [4544.42412451],
                [4042.5       ],
                [4745.42971888],
                [2238.40946844],
                [2798.5735786 ]]
        }

def get_distance_lst():
    ret_tuple_list = []
    ret_cnt = 0
    for k in pairwise_clusters_dict:
        v_lst = [v[0] for v in pairwise_clusters_dict[k]]
        v_lst.sort()
        ret_cnt += len(v_lst)
        for v in v_lst:
            ret_tuple_list.append((k,v))
    return ret_tuple_list, ret_cnt