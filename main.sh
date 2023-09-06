# train
#bash ./tools/dist_train.sh configs/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_Rui.py 1 --work-dir work_dirs/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_Rui/

#bash ./tools/dist_train.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k.py 1 --work-dir work_dirs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k/
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_Rui.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k/
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_cmp.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_cmp/


#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_debug.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_debug/

#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_finetune.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_finetune/
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_tiny.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_tiny/
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_tiny_cmp.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_tiny_cmp/


#bash ./tools/dist_train.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k_Rui.py 1 --work-dir work_dirs/restorers/haedsr/edsr_x4c64b16_g1_300k_div2k_Rui/

#bash ./tools/dist_train.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k_cmp.py 1 --work-dir work_dirs/restorers/haedsr/edsr_x4c64b16_g1_300k_div2k_cmp/

#bash ./tools/dist_train.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k_Rui.py 1 --work-dir work_dirs/restorers/haedsr/edsr_x4c64b16_g1_300k_div2k_aniso/

#bash ./tools/dist_train.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k_cmp.py 1 --work-dir work_dirs/restorers/haedsr/edsr_x4c64b16_g1_300k_div2k_aniso_cmp/

#bash ./tools/dist_train.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k_tiny.py 1 --work-dir work_dirs/restorers/haedsr/edsr_x4c64b16_g1_300k_div2k_tiny/
#bash ./tools/dist_train.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k_tiny_cmp.py 1 --work-dir work_dirs/restorers/haedsr/edsr_x4c64b16_g1_300k_div2k_tiny_cmp/
#bash ./tools/dist_train.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k_finetune.py 1 --work-dir work_dirs/restorers/haedsr/edsr_x4c64b16_g1_300k_div2k_finetune/

#bash ./tools/dist_train.sh configs/restorers/edsr/edsr_x4c64b16_div2k_contrastive_tiny.py 1 --work-dir work_dirs/restorers/haedsr/edsr_x4c64b16_div2k_contrastive_tiny/
#bash ./tools/dist_train.sh configs/restorers/edsr/edsr_x4c64b16_div2k_contrastive.py 1 --work-dir work_dirs/restorers/haedsr/edsr_x4c64b16_div2k_contrastive/
# iso
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2k_contrastive.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_div2k_contrastive/
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2k_contrastive_both.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_div2k_contrastive_both/
# aniso
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2k_contrastive_aniso.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_div2k_contrastive_aniso/
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2k_contrastive_aniso_both.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_div2k_contrastive_aniso_both/

# test
#bash ./tools/dist_test.sh configs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_cmp.py work_dirs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_cmp/iter_100000.pth 1 
#bash ./tools/dist_test.sh configs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_Rui.py work_dirs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k/iter_100000.pth 1
#bash ./tools/dist_test.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k_Rui.py work_dirs/restorers/haedsr/edsr_x4c64b16_g1_300k_div2k_Rui/iter_100000.pth 1
#bash ./tools/dist_test.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k_cmp.py work_dirs/restorers/haedsr/edsr_x4c64b16_g1_300k_div2k_cmp/iter_100000.pth 1

#bash ./tools/dist_test.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k_Rui.py work_dirs/restorers/haedsr/edsr_x4c64b16_g1_300k_div2k_aniso/iter_100000.pth 1
#bash ./tools/dist_test.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k_cmp.py work_dirs/restorers/haedsr/edsr_x4c64b16_g1_300k_div2k_aniso_cmp/iter_100000.pth 1

#bash ./tools/dist_test.sh configs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_finetune.py work_dirs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_finetune/iter_100000.pth 1
#bash ./tools/dist_test.sh configs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_tiny.py work_dirs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_tiny/iter_4000.pth 1
#bash ./tools/dist_test.sh configs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_tiny_cmp.py work_dirs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_tiny_cmp/iter_4000.pth 1
#bash ./tools/dist_test.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k_finetune.py work_dirs/restorers/haedsr/edsr_x4c64b16_g1_300k_div2k_finetune/iter_100000.pth 1
#bash ./tools/dist_test.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k_tiny_cmp.py work_dirs/restorers/haedsr/edsr_x4c64b16_g1_300k_div2k_tiny_cmp/iter_4000.pth 1
#bash ./tools/dist_test.sh configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k_tiny.py work_dirs/restorers/haedsr/edsr_x4c64b16_g1_300k_div2k_tiny/iter_4000.pth 1
#bash ./tools/dist_test.sh configs/restorers/edsr/edsr_x4c64b16_div2k_contrastive_tiny.py work_dirs/restorers/haedsr/edsr_x4c64b16_div2k_contrastive_tiny/iter_4000.pth 1

#bash ./tools/dist_test.sh configs/restorers/edsr/edsr_x4c64b16_div2k_contrastive_tiny.py work_dirs/restorers/haedsr/edsr_x4c64b16_div2k_contrastive_tiny/iter_4000.pth 1
#bash ./tools/dist_test.sh configs/restorers/edsr/edsr_x4c64b16_div2k_contrastive.py work_dirs/restorers/haedsr/edsr_x4c64b16_div2k_contrastive/iter_100000.pth 1
#bash ./tools/dist_test.sh configs/restorers/edsr/edsr_x4c64b16_div2k_contrastive.py work_dirs/restorers/haedsr/edsr_x4c64b16_div2k_contrastive/iter_100000.pth 1

#bash ./tools/dist_test.sh configs/restorers/dasr/dasr_x4c64b16_div2k_contrastive_both.py work_dirs/restorers/dasr/dasr_x4c64b16_div2k_contrastive_both/iter_100000.pth 1
#bash ./tools/dist_test.sh configs/restorers/dasr/dasr_x4c64b16_div2k_contrastive_aniso_both.py work_dirs/restorers/dasr/dasr_x4c64b16_div2k_contrastive_aniso_both/iter_100000.pth 1


# official experiments begin
# dataset: DIV2K + Flickr2K
# isotropic blurring + bicubic downsampling
# 1. train contrastive part
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive/
# 2. train the whole network
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_both.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_both_testcont0.05/
# test part
#bash ./tools/dist_test.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_both.py work_dirs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_both/iter_300000.pth 1 


bash ./tools/dist_test.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both.py work_dirs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both_6-layer-pretrain/iter_200000.pth 1 

# dataset: DIV2K + Flickr2K
# anisotropic blurring + bicubic downsampling
# 1. train contrastive part
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_aniso.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_aniso/
# 2. train the whole network
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_both_aniso.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_both_aniso/

# dataset: DIV2K + Flickr2K
# MOCO comparison
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_cmp/
# 2. train the whole network
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both/

# 05092022 continue pretrained 6-layer
bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both_6-layer-pretrain/

# test on RDN pretrain
bash ./tools/dist_test.sh configs/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_TestRui.py configs/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_20210419-3577d44f.pth 1
  # save the images
  bash ./tools/dist_test.sh configs/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_TestRui.py configs/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_20210419-3577d44f.pth 1 --save-path work_dirs/restorers/rdn/X4/Real-Micron/ 


# test on Real-ESRGAN pretrain
bash ./tools/dist_test.sh configs/restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost_Rui.py configs/restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost_20211010-34798885.pth 1
  # save the images
  bash ./tools/dist_test.sh configs/restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost_Rui_Ours.py configs/restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost_20211010-34798885.pth 1 --save-path work_dirs/restorers/real-esrgan/X4/Ours_noNew/

# 0627 continue on 2X
bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/X2/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both_6-layer-pretrain/
# add eval process to see the output
bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/X2/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both_6-layer-pretrain_eval/

# 0716 test 2X
bash ./tools/dist_test.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both.py work_dirs/restorers/dasr/X2/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both_6-layer-pretrain_aniso/iter_200000.pth 1

# 0915 test 2X
bash ./tools/dist_test.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both.py work_dirs/restorers/dasr/X2/dasr_x4c64b16_div2kflickr2k_contrastive_MoCo_both_6-layer-pretrain/iter_200000.pth 1

# test on Real-ESRGAN pretrain 2X
bash ./tools/dist_test.sh configs/restorers/real_esrgan/.py configs/restorers/real_esrgan/.pth 1

# test on RDN pretrain 2X
bash ./tools/dist_test.sh configs/restorers/rdn/rdn_x2c64b16_g1_1000k_div2k_TestRui.py configs/restorers/rdn/rdn_x2c64b16_g1_1000k_div2k_20210419-dc146009.pth 1

bash ./tools/dist_test.sh configs/restorers/rdn/rdn_x2c64b16_g1_1000k_div2k_TestRui.py configs/restorers/rdn/rdn_x2c64b16_g1_1000k_div2k_20210419-dc146009.pth 1 --save-path work_dirs/restorers/rdn/X2/DRealSR/ 


# 09192022 Train 2X on SimCLR and 6-layer
bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_both.py 1 --work-dir work_dirs/restorers/dasr/X2/dasr_x4c64b16_div2kflickr2k_contrastive_SimCLR_both/

# 09222022 Test 2X on SimCLR and 6-layer
bash ./tools/dist_test.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_both.py work_dirs/restorers/dasr/X2/dasr_x4c64b16_div2kflickr2k_contrastive_SimCLR_both/iter_200000.pth 1

# 20221107 try to deal with our data
# dataset: ours
# 1. train contrastive part
#bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_div2kflickr2k_contrastive_aniso.py 1 --work-dir work_dirs/restorers/dasr/testtesttest/

bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_ours_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/X2/transfer_ours_contrastive/ 

# 2. train the whole network
bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_ours_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/X2/transfer_ours_frozen4/ 
bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_ours_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/X2/transfer_ours_frozen3/ 
bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_ours_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/X2/transfer_ours_frozen2/ 
bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_ours_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/X2/transfer_ours_frozen0/ 

# adjust cropping and lr and gradient clipping
bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_ours_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/X2/transfer_ours_frozen0_crop112/
bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_ours_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/X2/transfer_ours_frozen1_crop112/
bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_ours_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/X2/transfer_ours_frozen2_crop112/
bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_ours_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/X2/transfer_ours_frozen3_crop112/
bash ./tools/dist_train.sh configs/restorers/dasr/dasr_x4c64b16_ours_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/X2/transfer_ours_frozen4_crop112/
# test X2 our data 20221201
bash ./tools/dist_test.sh configs/restorers/dasr/dasr_x4c64b16_ours_contrastive_MoCo_both.py work_dirs/restorers/dasr/X2/transfer_ours_frozen0_crop112/iter_40000.pth 1

#X4 20221130

bash ./tools/dist_train.sh configs/restorers/dasr/dasrX4_x4c64b16_ours_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/dasr/X4/transfer_ours_frozen0_crop192/


# get flops
python tools/get_flops.py configs/restorers/dasr/dasr_x4c64b16_ours_contrastive_MoCo_both.py --shape 96 96

python tools/get_flops.py configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py --shape 96 96



# inference images
python demo/restoration_demo.py configs/restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost_Rui.py configs/restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost_20211010-34798885.pth /home/rui/Rui_registration/datasets/testdata2_part/0019.tif demo/0019_SR.png

