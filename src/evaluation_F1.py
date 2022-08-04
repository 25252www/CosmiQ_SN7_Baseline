# 重写solaris.eval.challenges.spacenet_buildings_2 方法
import pandas as pd
import solaris as sol

def evaluation():
    report_csv = '/home/liuxiangyu/CosmiQ_SN7_Baseline/sn7_baseline_report.csv'
    truth_csv = '/home/liuxiangyu/CosmiQ_SN7_Baseline/sn7_baseline_labels.csv'
    prop_csv = '/home/liuxiangyu/CosmiQ_SN7_Baseline/inference_out/sn7_baseline_preds/csvs/sn7_baseline_predictions.csv'
    truth_csv_standard = '/home/liuxiangyu/CosmiQ_SN7_Baseline/sn7_baseline_labels_standard.csv'
    prop_csv_standard = '/home/liuxiangyu/CosmiQ_SN7_Baseline/inference_out/sn7_baseline_preds/csvs/sn7_baseline_predictions_standard.csv'
    # 看看预测结果和标签都有多少个，修改列名
    proposaldf=pd.read_csv(prop_csv)
    num_proposal = len(proposaldf['filename'].unique())
    print('The number of images in proposal:{}'.format(num_proposal))
    proposaldf = proposaldf.rename(columns={'geometry':'PolygonWKT_Pix'})
    proposaldf = proposaldf.rename(columns={'id':'BuildingId'})
    proposaldf = proposaldf.rename(columns={'filename':'ImageId'})
    proposaldf.to_csv(prop_csv_standard)
    
    truthdf = pd.read_csv(truth_csv)
    num_truth = len(truthdf['filename'].unique())
    print('The number of images in ground_truth: {}'.format(num_truth))
    truthdf = truthdf.rename(columns={'geometry':'PolygonWKT_Pix'})
    truthdf = truthdf.rename(columns={'id':'BuildingId'})
    truthdf = truthdf.rename(columns={'filename':'ImageId'})
    truthdf.to_csv(truth_csv_standard)

    evaluator = sol.eval.base.Evaluator(truth_csv_standard)
    evaluator.load_proposal(prop_csv_standard, proposalCSV=True, conf_field_list=[])
    report = evaluator.eval_iou_spacenet_csv(miniou=0.25, min_area=4.)
    
    #eval_iou_spacenet_csv的结果保存到csv中
    li = [entry for entry in report]
    reportdf = pd.DataFrame(li,columns=['imageID', 'iou_field', 'TruePos', 'FalsePos', 'FalseNeg',
                'Precision', 'Recall', 'F1Score'])
    reportdf.to_csv(report_csv, index=False)

    tp = 0
    fp = 0
    fn = 0
    for entry in report:
        tp += entry['TruePos']
        fp += entry['FalsePos']
        fn += entry['FalseNeg']
    f1score = (2*tp) / (2*tp + fp + fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print('Vector F1: {}'.format(f1score))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))

evaluation()
