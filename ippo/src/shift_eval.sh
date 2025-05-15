#!/bin/bash

# declare -a paths=(\
    # "results_coco2/models/20220930_015224_ia2c_ns_perturb_minimal_seed649181501" \
    # "results_coco2/models/20220930_015229_ia2c_ns_msg_perturb_minimal_seed723759704" \
    # "results_coco2/models/20220930_015235_ia2c_ns_msg_conf_perturb_minimal_seed780419442" \
    # "results_coco2/models/20220930_015243_ia2c_ns_msg_conf_ext_perturb_minimal_seed751625057" \
    # "results_coco2/models/20220930_015249_ia2c_ns_sd_conf_perturb_minimal_seed988060419" \
    # "results_coco2/models/20220930_015254_ia2c_ns_msg_fusion_perturb_minimal_seed75482834" \
# )


# for path in "${paths[@]}"; do
#     python src/main.py --config=ia2c_ns --env-config=rware with checkpoint_path="$path"
# done


for i in {1..10}; do
    echo "*** $i"
    echo "*** ia2c ns"
    python src/main.py --config=ia2c_ns --env-config=rware with checkpoint_path="results_v1/models/20221001_013300_ia2c_ns_perturb_distshift-final_seed838431072"
    python src/main.py --config=ia2c_ns --env-config=rware with checkpoint_path="results_v1/models/20221001_124231_ia2c_ns_perturb_distshift-final_seed651994714"
    python src/main.py --config=ia2c_ns --env-config=rware with checkpoint_path="results_coco2/models/20221001_012737_ia2c_ns_perturb_distshift-final_seed310083790"
    python src/main.py --config=ia2c_ns --env-config=rware with checkpoint_path="results_coco2/models/20221001_094843_ia2c_ns_perturb_distshift-final_seed820610646"
    python src/main.py --config=ia2c_ns --env-config=rware with checkpoint_path="results_coco2/models/20221001_184118_ia2c_ns_perturb_distshift-final_seed53362479"
    echo "*** extr"
    python src/main.py --config=ia2c_ns_msg --env-config=rware with checkpoint_path="results_v1/models/20221001_013308_ia2c_ns_msg_perturb_distshift-final_seed147609242"
    python src/main.py --config=ia2c_ns_msg --env-config=rware with checkpoint_path="results_v1/models/20221001_124235_ia2c_ns_msg_perturb_distshift-final_seed376472977"
    python src/main.py --config=ia2c_ns_msg --env-config=rware with checkpoint_path="results_coco2/models/20221001_012745_ia2c_ns_msg_perturb_distshift-final_seed12725300"
    python src/main.py --config=ia2c_ns_msg --env-config=rware with checkpoint_path="results_coco2/models/20221001_094852_ia2c_ns_msg_perturb_distshift-final_seed908638502"
    python src/main.py --config=ia2c_ns_msg --env-config=rware with checkpoint_path="results_coco2/models/20221001_184120_ia2c_ns_msg_perturb_distshift-final_seed322525365"
    echo "*** ues"
    python src/main.py --config=ia2c_ns_sd_conf --env-config=rware with checkpoint_path="results_v1/models/20221001_013335_ia2c_ns_sd_conf_perturb_distshift-final_seed72948959"
    python src/main.py --config=ia2c_ns_sd_conf --env-config=rware with checkpoint_path="results_v1/models/20221001_124246_ia2c_ns_sd_conf_perturb_distshift-final_seed862472640"
    python src/main.py --config=ia2c_ns_sd_conf --env-config=rware with checkpoint_path="results_coco2/models/20221001_012809_ia2c_ns_sd_conf_perturb_distshift-final_seed512800294"
    python src/main.py --config=ia2c_ns_sd_conf --env-config=rware with checkpoint_path="results_coco2/models/20221001_094901_ia2c_ns_sd_conf_perturb_distshift-final_seed12615220"
    python src/main.py --config=ia2c_ns_sd_conf --env-config=rware with checkpoint_path="results_coco2/models/20221001_184128_ia2c_ns_sd_conf_perturb_distshift-final_seed162302448"
    echo "*** ues+extr"
    python src/main.py --config=ia2c_ns_msg_fusion --env-config=rware with checkpoint_path="results_v1/models/20221001_013341_ia2c_ns_msg_fusion_perturb_distshift-final_seed710495348"
    python src/main.py --config=ia2c_ns_msg_fusion --env-config=rware with checkpoint_path="results_v1/models/20221001_124250_ia2c_ns_msg_fusion_perturb_distshift-final_seed389057405"
    python src/main.py --config=ia2c_ns_msg_fusion --env-config=rware with checkpoint_path="results_coco2/models/20221001_012817_ia2c_ns_msg_fusion_perturb_distshift-final_seed918981122"
    python src/main.py --config=ia2c_ns_msg_fusion --env-config=rware with checkpoint_path="results_coco2/models/20221001_094905_ia2c_ns_msg_fusion_perturb_distshift-final_seed490127542"
    python src/main.py --config=ia2c_ns_msg_fusion --env-config=rware with checkpoint_path="results_coco2/models/20221001_184129_ia2c_ns_msg_fusion_perturb_distshift-final_seed174735547"
    echo "*** mappo"
    python src/main.py --config=mappo --env-config=rware with checkpoint_path="results_v1/models/20221001_013351_mappo_perturb_distshift-final_seed666566254"
    python src/main.py --config=mappo --env-config=rware with checkpoint_path="results_v1/models/20221001_124255_mappo_perturb_distshift-final_seed343254875"
    python src/main.py --config=mappo --env-config=rware with checkpoint_path="results_coco2/models/20221001_012912_mappo_perturb_distshift-final_seed890692028"
    python src/main.py --config=mappo --env-config=rware with checkpoint_path="results_coco2/models/20221001_094910_mappo_perturb_distshift-final_seed786305297"
    python src/main.py --config=mappo --env-config=rware with checkpoint_path="results_coco2/models/20221001_184133_mappo_perturb_distshift-final_seed490208678"
done
