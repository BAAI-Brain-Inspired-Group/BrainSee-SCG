CUDA_VISIBLE_DEVICES=1 python ./hyper2img_test.py --cur_config './models/cldm_v15_0_256_tune.yaml' --cur_ckpt './controlNet_ckpt/ckp_0_512_few1/last.ckpt' --output_dir './output' --test_json_path './data/testdata.json' --hyper_scale [1]
python ./train.py --modelarch_path './models/cldm_v15_0_256_tune.yaml' --checkpoint_path 'ckp_0_512_few1' --logger_path 'res_0_512_few1'  --resume_path './models/control_sd15_ini.ckpt' --dataset_name 'MyDataset_Hazy_FewShot'


CUDA_VISIBLE_DEVICES=1 python ./hyper2img_test.py --cur_config './models/cldm_v15_0_256_tune.yaml' --cur_ckpt ./controlNet_ckpt/ckp_0_512_few1/last.ckpt --output_dir './output' --test_json_path './data/testdata.json' --hyper_scale [1]
python ./train.py --modelarch_path './models/cldm_v15_0_256_tune.yaml' --checkpoint_path 'ckp_0_512_few1' --logger_path 'res_0_512_few1'  --resume_path './multi_ckpt/epoch=10-step=325291.ckpt' --dataset_name 'MyDataset_Hazy_FewShot'

CUDA_VISIBLE_DEVICES=1 python ./hyper2img_test.py --cur_config './models/cldm_v15_0_512.yaml' --cur_ckpt ./controlNet_ckpt/ckp_0_512_few1_new/last.ckpt --output_dir './output' --test_json_path './data/testdata.json' --hyper_scale [1]
python ./train.py --modelarch_path './models/cldm_v15_0_512.yaml' --checkpoint_path 'ckp_0_512_few1' --logger_path 'res_0_512_few1'  --resume_path './multi_ckpt/epoch=10-step=325291.ckpt' --dataset_name 'MyDataset_Hazy_FewShot'

CUDA_VISIBLE_DEVICES=1 python ./hyper2img_test_0.py --cur_config './models/cldm_v15_0_512.yaml' --cur_ckpt './multi_ckpt/epoch=10-step=325291.ckpt' --output_dir './output' --test_json_path './data/testdata.json' --hyper_scale [1]
python ./hyper2img_test_0_256.py --cur_config './models/cldm_v15_0_256.yaml' --cur_ckpt './multi_ckpt/epoch=10-step=325291.ckpt' --output_dir './output' --test_json_path './data/testdata.json' --hyper_scale [1]

CUDA_VISIBLE_DEVICES=1 python ./hyper2img_test_0_0.25_0.75.py --cur_config './models/cldm_v15_0_512.yaml' --cur_ckpt './multi_ckpt/0_epoch=10-step=325291.ckpt' --output_dir './output' --test_json_path './data/testdata.json' --hyper_scale [1]
CUDA_VISIBLE_DEVICES=1 python ./hyper2img_test_0_0.5_0.75.py --cur_config './models/cldm_v15_0_512.yaml' --cur_ckpt './multi_ckpt/0_epoch=10-step=325291.ckpt' --output_dir './output' --test_json_path './data/testdata.json' --hyper_scale [1]
CUDA_VISIBLE_DEVICES=1 python ./hyper2img_test_0_0.75_0.75.py --cur_config './models/cldm_v15_0_512.yaml' --cur_ckpt './multi_ckpt/0_epoch=10-step=325291.ckpt' --output_dir './output' --test_json_path './data/testdata.json' --hyper_scale [1]

python ./train.py --modelarch_path './models/cldm_v15_1_512.yaml' --checkpoint_path 'ckp_1_512_few1' --logger_path 'res_1_512_few1'  --resume_path './multi_ckpt/1_epoch=10-step=325291.ckpt' --dataset_name 'MyDataset_Hazy_FewShot'

CUDA_VISIBLE_DEVICES=0 python ./hyper2img_test_1_0.5_0.5.py --cur_config './models/cldm_v15_0_512.yaml' --cur_ckpt './controlNet_ckpt/ckp_0_512_few1/last.ckpt' --output_dir './output' --test_json_path './data/testdata.json' --hyper_scale [1]
CUDA_VISIBLE_DEVICES=0 python ./hyper2img_test_1_0.75_0.75.py --cur_config './models/cldm_v15_0_512.yaml' --cur_ckpt './controlNet_ckpt/ckp_0_512_few1/last.ckpt' --output_dir './output' --test_json_path './data/testdata.json' --hyper_scale [1]
CUDA_VISIBLE_DEVICES=0 python ./hyper2img_test_1_1_1.py --cur_config './models/cldm_v15_0_512.yaml' --cur_ckpt './controlNet_ckpt/ckp_0_512_few1/last.ckpt' --output_dir './output' --test_json_path './data/testdata.json' --hyper_scale [1]

/home/uchihawdt/miniconda3/envs/cont/bin/python /home/uchihawdt/control-net-main-v0.5/1.py --gt_dir /home/uchihawdt/DehazeFormer/data/RESIDE-IN/test/GT --gen_dir /home/uchihawdt/control-net-main-v0.5/output/few0_0.75_0.5

python ./train.py --modelarch_path './models/cldm_v15_0_512.yaml' --checkpoint_path 'ckp_0_512_few1_light' --logger_path 'res_0_512_few1_light'  --resume_path './multi_ckpt/0_epoch=10-step=325291.ckpt' --dataset_name 'MyDataset_Light_FewShot'
python ./train.py --modelarch_path './models/cldm_v15_1_512.yaml' --checkpoint_path 'ckp_1_512_few1_light' --logger_path 'res_1_512_few1_light'  --resume_path './multi_ckpt/1_epoch=10-step=325291.ckpt' --dataset_name 'MyDataset_Light_FewShot'

CUDA_VISIBLE_DEVICES=1 python ./hyper2img_test_0_0.25_0.75.py --cur_config './models/cldm_v15_0_512.yaml' --cur_ckpt './multi_ckpt/0_epoch=10-step=325291.ckpt' --output_dir './output' --test_json_path './data/testdata.json' --hyper_scale [1]
CUDA_VISIBLE_DEVICES=1 python ./hyper2img_test_0_0.5_0.75.py --cur_config './models/cldm_v15_0_512.yaml' --cur_ckpt './multi_ckpt/0_epoch=10-step=325291.ckpt' --output_dir './output' --test_json_path './data/testdata.json' --hyper_scale [1]
CUDA_VISIBLE_DEVICES=1 python ./hyper2img_test_0_0.75_0.75.py --cur_config './models/cldm_v15_0_512.yaml' --cur_ckpt './multi_ckpt/0_epoch=10-step=325291.ckpt' --output_dir './output' --test_json_path './data/testdata.json' --hyper_scale [1]