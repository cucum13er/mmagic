# X4 HASR DRealSR
bash ./tools/dist_train.sh configs/restorers/edsr/haedsr_contrastive_DRealSR.py 1 --work-dir work_dirs/restorers/edsr/X4/haedsr_initial/ 

bash ./tools/dist_test.sh configs/restorers/edsr/haedsr_contrastive_DRealSR.py work_dirs/restorers/edsr/X4/haedsr_initial/iter_80000.pth 1

# ablation
# set lambda~=0, lr=1e-19
bash ./tools/dist_train.sh configs/restorers/edsr/haedsr_contrastive_DRealSR.py 1 --work-dir work_dirs/restorers/edsr/X4/haedsr_ablation/ 

bash ./tools/dist_test.sh configs/restorers/edsr/haedsr_contrastive_DRealSR.py work_dirs/restorers/edsr/X4/haedsr_ablation/iter_80000.pth 1





