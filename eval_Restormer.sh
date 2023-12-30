# X4 DIV2K pretrain
bash ./tools/dist_train.sh configs/restorers/restormer/restormer_Rui_original_DIV2K_X4.py 1 --work-dir work_dirs/restorers/restormer/X4/pretrain_DIV2K/ 
# X4 DIV2K test pretrain
bash ./tools/dist_test.sh configs/restorers/restormer/restormer_Rui_original_DIV2K_X4.py work_dirs/restorers/restormer/X4/pretrain_DIV2K/iter_540.pth 1

# X4 DIV2K Channel Attention after Transformer block
bash ./tools/dist_train.sh configs/restorers/restormer/restormer_Rui_CA_DIV2K_X4.py 1 --work-dir work_dirs/restorers/restormer/X4/CA_DIV2K/ 
  # test results
  bash ./tools/dist_test.sh configs/restorers/restormer/restormer_Rui_CA_DIV2K_X4.py work_dirs/restorers/restormer/X4/CA_DIV2K/iter_160000.pth 1 --save-path work_dirs/restorers/restormer/X4/CA_DIV2K/Set5_4.0/
 #/work/pi_xiandu_umass_edu/ruima/git/mmediting_Rui_git/configs/restorers/restormer/restormer_Rui_CA_DIV2K_X4.py
# X4 DRealSR pretrain
bash ./tools/dist_train.sh configs/restorers/restormer/restormer_Rui_original_DRealSR_X4.py 1 --work-dir work_dirs/restorers/restormer/X4/pretrain_DRealSR/ 
  # test results
  bash ./tools/dist_test.sh configs/restorers/restormer/restormer_Rui_original_DRealSR_X4.py work_dirs/restorers/restormer/X4/pretrain_DRealSR/iter_2000.pth 1

# X4 DRealSR Channel Attention after Transformer block
bash ./tools/dist_train.sh configs/restorers/restormer/restormer_Rui_CA_DRealSR_X4.py 1 --work-dir work_dirs/restorers/restormer/X4/CA_DRealSR/ 
  # test results
  bash ./tools/dist_test.sh configs/restorers/restormer/restormer_Rui_CA_DRealSR_X4.py work_dirs/restorers/restormer/X4/CA_DRealSR/iter_200000.pth 1 --save-path work_dirs/restorers/restormer/X4/CA_DRealSR/DRealSR/

# X2 DIV2K pretrain
bash ./tools/dist_train.sh configs/restorers/restormer/restormer_Rui_original_DIV2K_X2.py 1 --work-dir work_dirs/restorers/restormer/X2/pretrain_DIV2K/ 
  # test result
  bash ./tools/dist_test.sh configs/restorers/restormer/restormer_Rui_original_DIV2K_X2.py work_dirs/restorers/restormer/X2/pretrain_DIV2K/iter_1000.pth 1

# X2 DRealSR pretrain
bash ./tools/dist_train.sh configs/restorers/restormer/restormer_Rui_original_DRealSR_X2.py 1 --work-dir work_dirs/restorers/restormer/X2/pretrain_DRealSR/ 
  # test result
  bash ./tools/dist_test.sh configs/restorers/restormer/restormer_Rui_original_DRealSR_X2.py work_dirs/restorers/restormer/X2/pretrain_DRealSR/iter_2000.pth 1

# X2 DIV2K Channel Attention after Transformer block
bash ./tools/dist_train.sh configs/restorers/restormer/restormer_Rui_CA_DIV2K_X2.py 1 --work-dir work_dirs/restorers/restormer/X2/CA_DIV2K/ 
  # test result
  bash ./tools/dist_test.sh configs/restorers/restormer/restormer_Rui_CA_DIV2K_X2.py work_dirs/restorers/restormer/X2/CA_DIV2K/iter_120000.pth 1

# X2 DRealSR Channel Attention after Transformer block
bash ./tools/dist_train.sh configs/restorers/restormer/restormer_Rui_CA_DRealSR_X2.py 1 --work-dir work_dirs/restorers/restormer/X2/CA_DRealSR/ 
  # test result
  bash ./tools/dist_test.sh configs/restorers/restormer/restormer_Rui_CA_DRealSR_X2.py work_dirs/restorers/restormer/X2/CA_DRealSR/iter_180000.pth 1


# showing full name of my tasks
squeue --format="%.18i %.15P %.40j %.8u %.8T %.10M %.9l %.6D %R" --me




'''
# X4 HASR div2k_flickr2k
bash ./tools/dist_train.sh configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui.py 1 --work-dir work_dirs/restorers/swinir/X4/swinir_initial/ 

bash ./tools/dist_test.sh configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui.py SwinIR_work_dir/iter_80000.pth 1

# save images
bash ./tools/dist_test.sh configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui_basic.py work_dirs/restorers/swinir/X4/swinir_basic/iter_80000.pth 1 --save-path work_dirs/restorers/swinir/X4/swinir_basic/sr_results/Urban100_4.0/

bash ./tools/dist_test.sh configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui.py SwinIR_work_dir/iter_80000.pth 1 --save-path SwinIR_work_dir/sr_results/Urban100_sig4.0/


# X2 HASR div2k_flickr2k
bash ./tools/dist_train.sh configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui_basicX2.py 1 --work-dir work_dirs/restorers/swinir/X2/basic/ 

bash ./tools/dist_test.sh  configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui_basicX2.py work_dirs/restorers/swinir/X2/basic/iter_80000.pth 1


# X4 DRealSR

bash ./tools/dist_train.sh configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui_DRealSR.py 1 --work-dir work_dirs/restorers/swinir/X4/swinir_DRealSR_basic/ 

bash ./tools/dist_test.sh configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui_DRealSR.py work_dirs/restorers/swinir/X4/swinir_DRealSR_basic/iter_180000.pth 1

bash ./tools/dist_test.sh configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui_DRealSR.py work_dirs/restorers/swinir/X4/swinir_DRealSR_basic/iter_180000.pth 1 --save-path work_dirs/restorers/swinir/X4/swinir_DRealSR_basic/sr_results/round1/


# X2 DRealSR

bash ./tools/dist_train.sh configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui_DRealSR_x2.py 1 --work-dir work_dirs/restorers/swinir/X2/swinir_DRealSR_basic_x2/ 





#############################################################################################################################################################


bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_1616/

bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_0808/
# X2 HASR div2k_flickr2k
bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/hasr/X2/hasr_initial/ 

'''



