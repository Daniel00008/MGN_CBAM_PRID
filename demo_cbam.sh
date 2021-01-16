python3 main.py --reset --datadir ../dataset/data_PRCV2020/ --batchid 8 --batchtest 32 --test_every 2 --epochs 4 --decay_type step_1_2 --loss 1*CrossEntropy+2*Triplet --margin 0.3 --re_rank --random_erasing --save MGN_CBAM_adam --cpu  --lr 2e-4 --optimizer ADAM --model MGN_SE
# python3 main.py --reset --datadir ../dataset/data_PRCV2020/ --batchid 8 --batchtest 32 --test_every 2 --epochs 4 --decay_type step_1_2 --loss 1*CrossEntropy+2*Triplet --margin 0.3 --re_rank --random_erasing --save MGN_CBAM_adam --cpu  --lr 2e-4 --optimizer ADAM --model MGN_CBAM2



