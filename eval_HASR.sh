# X4 HASR div2k_flickr2k
bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_initial/ 

bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_1616/

bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_0808/
# X2 HASR div2k_flickr2k
bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both.py 1 --work-dir work_dirs/restorers/hasr/X2/hasr_initial/ 


# Test
# X4
bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py work_dirs/restorers/hasr/X4/hasr_initial/iter_200000.pth 1

#################################################################################################################################################################################
# new 2023-10-05
# change the backbone /work/pi_xiandu_umass_edu/ruima/git/mmediting_Rui_git/mmedit/models/backbones/sr_backbones/hasr.py
bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py work_dirs/restorers/hasr_ca/X4/multi_gpus/iter_20000.pth 1

bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py work_dirs/restorers/hasr_sa/X4/multi_gpus/iter_20000.pth 1

# outside RCAB method
bash ./tools/dist_test.sh configs/restorers/hasr_single/hasr_single_div2kflickr2k_contrastive_MoCo_both_X4.py work_dirs/restorers/hasr_single/X4/multi_gpus/iter_20000.pth 1

#################################################################################################################################################################################



bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py work_dirs/restorers/hasr/X4/hasr_0808/iter_200000.pth 1 --save-path work_dirs/restorers/hasr/X4/hasr_0808/sr_results/Urban100_1.0/

# X2
bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both.py work_dirs/restorers/hasr/X2/hasr_initial/iter_100000.pth 1



# X2 HASR Real-Micron
# run 80000 iters already
bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_Real-Micron.py 1 --work-dir work_dirs/restorers/hasr/X2/hasr_real-micron_from_None/ 

# X4 HASR Real-Micron
#### runned on the other computer#############
bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_Real-Micron_X4.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_real-micron_from_None/ 
###################
# test Real-Micron
# X4
bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_Real-Micron_X4.py work_dirs/restorers/hasr/X4/transfer_hasr_0808_frozen2/iter_20000.pth 1 --save-path work_dirs/restorers/hasr/X4/transfer_hasr_0808_frozen2/sr_results/

# frozen2 is the best
bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_Real-Micron_X4.py work_dirs/restorers/hasr/X4/transfer_hasr_0808_frozen1/iter_20000.pth 1 --save-path work_dirs/restorers/hasr/X4/transfer_hasr_0808_frozen1/sr_results/C4112

# X2
bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_Real-Micron.py work_dirs/restorers/hasr/X2/hasr_real-micron_from_None/iter_200000.pth 1 --save-path work_dirs/restorers/hasr/X2/hasr_real-micron_from_None/sr_results/C640

# DRealSR experiments
# X4
bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_DRealSR_X4.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_0808_DRealSR/ 

''' DRealSR training, other datasets testing, not good
# X4 testing
bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_DRealSR_X4.py work_dirs/restorers/hasr/X4/hasr_0808_DRealSR/iter_200000.pth 1 --save-path work_dirs/restorers/hasr/X4/hasr_0808_DRealSR/sr_results/Urban100_4.0/
# use DRealSR test other datasets
bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_DRealSR_X4.py work_dirs/restorers/hasr/X4/hasr_0808_DRealSR/iter_200000.pth 1
'''
# DRealSR experiments
# X2
bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_DRealSR.py 1 --work-dir work_dirs/restorers/hasr/X2/hasr_0808_DRealSR/ 

bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_DRealSR.py work_dirs/restorers/hasr/X2/hasr_0808_DRealSR/iter_200000.pth 1
# save images
bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_DRealSR.py work_dirs/restorers/hasr/X2/hasr_0808_DRealSR/iter_200000.pth 1 --save-path work_dirs/restorers/hasr/X2/hasr_0808_DRealSR/sr_results/

bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_DRealSR_inference.py work_dirs/restorers/hasr/X2/hasr_0808_DRealSR/iter_200000.pth 1 --save-path work_dirs/restorers/hasr/X2/hasr_0808_DRealSR/testdata2_SR/
'''
not yet
# X2 testing
bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_DRealSR_X4.py work_dirs/restorers/hasr/X4/hasr_0808_DRealSR/iter_200000.pth 1
'''
# ablation study on DRealSR
# X4
bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_DRealSR_X4_ablation.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_0808_DRealSR_ablation/ 

bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_DRealSR_X4_ablation.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_0808_DRealSR_ablation2/ 

bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_DRealSR_X4_ablation.py work_dirs/restorers/hasr/X4/hasr_0808_DRealSR_ablation/iter_80000.pth 1
bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_DRealSR_X4_ablation.py work_dirs/restorers/hasr/X4/hasr_0808_DRealSR_ablation2/iter_200000.pth 1
# Start HASR experiments on Simulation data
# dataset: DIV2K + Flickr2K
bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_0808_fromX2pretrain/
#not run yet
#bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_0808/
#bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_single_0808/

# test
bash ./tools/dist_test.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py work_dirs/restorers/hasr/X4/hasr_0808_fromX2pretrain/iter_60000.pth 1




bash ./tools/slurm_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_0808_fromX2pretrain/


