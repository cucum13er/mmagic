# Reformer with only L1 loss 
# data_root='/media/rui/Samsung4TB/Datasets/testdata2_choose/'
# pre-registration 
# epoch 2000
bash ./tools/dist_test.sh configs/reformer/reformer_temp.py /home/rui/Rui_SR/mmagic_Rui/work_dirs/X2/iter_2000.pth 1
Results: 08/03 16:55:41 - mmengine - INFO - Iter(test) [26/26]    MAE: 0.0435  PSNR: 24.5568  SSIM: 0.3393  data_time: 0.0091  time: 0.0656
# epoch 3000
Results: 08/03 16:57:29 - mmengine - INFO - Iter(test) [26/26]    MAE: 0.0320  PSNR: 27.0874  SSIM: 0.4648  data_time: 0.0092  time: 0.0654

# using EDSR pretrained to test the results
# using ENV: openmmlab
bash ./tools/dist_test.sh configs/restorers/edsr/edsr_x2c64b16_g1_300k_div2k_eval_reformer.py configs/restorers/edsr/edsr_x2c64b16_1x16_300k_div2k_20200604-19fe95ea.pth 1
Results: Eval-PSNR: 28.847156186655695 Eval-SSIM: 0.6239148379148841



