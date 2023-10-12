# X4 HASR div2k_flickr2k
bash ./tools/dist_train.sh configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui.py 1 --work-dir work_dirs/restorers/swinir/X4/swinir_initial/ 

bash ./tools/dist_test.sh configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui.py work/pi_xiandu_umass_edu/ruima/git/mmediting_Rui_git/SwinIR_work_dir/iter_80000.pth 1

# X4 DRealSR

bash ./tools/dist_train.sh configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui_DRealSR.py 1 --work-dir work_dirs/restorers/swinir/X4/swinir_DRealSR_basic/ 

# X2 DRealSR

bash ./tools/dist_train.sh configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui_DRealSR_x2.py 1 --work-dir work_dirs/restorers/swinir/X2/swinir_DRealSR_basic_x2/ 


#############################################################################################################################################################


bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_1616/

bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_0808/
# X2 HASR div2k_flickr2k
bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/hasr/X2/hasr_initial/ 







