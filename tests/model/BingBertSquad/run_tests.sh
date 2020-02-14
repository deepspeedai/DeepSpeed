#!/bin/bash

if [[ ! -d logs ]]
then
        mkdir logs
fi

validate_file() {
  file=$1
  file_name=$2

  if [[ -f $file ]]; then
    echo "Using ${file_name}: ${file}"
  else
    echo "${file} not found"
    exit 1
  fi
}

validate_folder() {
	dir=$1
	dir_name=$2

	if [[ -d ${dir} ]]; then
		echo "Using ${dir_name}: ${dir}"
	else
		echo "${dir} folder not found"
		exit 1
	fi
}

# Validate path to BingBertSquad script
if [ -z "${BingBertSquad_DIR+x}" ]; then
  export BingBertSquad_DIR=../../../DeepSpeedExamples/BingBertSquad
  echo "BingBertSquad_DIR environment variable not set; trying default: ${BingBertSquad_DIR}"
fi
validate_folder ${BingBertSquad_DIR} "BingBertSquad_DIR"

fp16_config_json=deepspeed_bsz24_fp16_config.json
validate_file ${fp16_config_json} "fp16_config_json"
fp32_config_json=deepspeed_bsz24_fp32_config.json
validate_file ${fp32_config_json} "fp32_config_json"

start_time=`date +"%D %T"`
echo "---------------begin @ ${start_time}--------------"
# Note: you may play around with commented parts below (num_gpus and nohup command) for simultaneous runs; just make sure your hardware allocation can support it
for num_gpus in 8 1 # 4 2
do
        #run_cmd="nohup bash run_BingBertSquad.sh -g ${num_gpus} -d --deepspeed_config ${fp16_config_json} --fp16 > logs/deepspeed_fp16_${num_gpus}_`date +"%Y%m%d%H%M%S"`.out 2> logs/deepspeed_fp16_${num_gpus}_`date +"%Y%m%d%H%M%S"`.err &"
        run_cmd="bash run_BingBertSquad.sh -g ${num_gpus} -d --deepspeed_config ${fp16_config_json} --fp16"
        start_time=`date +"%D %T"`
        echo "---------------begin @ ${start_time}--------------"
        echo ${run_cmd}
        eval ${run_cmd}
        end_time=`date +"%D %T"`
        echo "---------------finish @ ${end_time} --------------"

        #run_cmd="nohup bash run_BingBertSquad.sh -g ${num_gpus} -d --deepspeed_config ${fp32_config_json} > logs/deepspeed_fp32_${num_gpus}_`date +"%Y%m%d%H%M%S"`.out 2> logs/deepspeed_fp32_${num_gpus}_`date +"%Y%m%d%H%M%S"`.err &"
        run_cmd="bash run_BingBertSquad.sh -g ${num_gpus} -d --deepspeed_config ${fp32_config_json}"
        start_time=`date +"%D %T"`
        echo "---------------begin @ ${start_time}--------------"
        echo ${run_cmd}
        eval ${run_cmd}
        end_time=`date +"%D %T"`
        echo "---------------finish @ ${end_time} --------------"

        #run_cmd="nohup bash run_BingBertSquad.sh -g ${num_gpus} --fp16 > logs/baseline_fp16_${num_gpus}_`date +"%Y%m%d%H%M%S"`.out 2> logs/baseline_fp16_${num_gpus}_`date +"%Y%m%d%H%M%S"`.err &"
        run_cmd="bash run_BingBertSquad.sh -g ${num_gpus} --fp16"
        start_time=`date +"%D %T"`
        echo "---------------begin @ ${start_time}--------------"
        echo ${run_cmd}
        eval ${run_cmd}
        end_time=`date +"%D %T"`
        echo "---------------finish @ ${end_time} --------------"

        #run_cmd="nohup bash run_BingBertSquad.sh -g ${num_gpus} > logs/baseline_fp32_${num_gpus}_`date +"%Y%m%d%H%M%S"`.out 2> logs/baseline_fp32_${num_gpus}_`date +"%Y%m%d%H%M%S"`.err &"
        run_cmd="bash run_BingBertSquad.sh -g ${num_gpus}"
        start_time=`date +"%D %T"`
        echo "---------------begin @ ${start_time}--------------"
        echo ${run_cmd}
        eval ${run_cmd}
        end_time=`date +"%D %T"`
        echo "---------------finish @ ${end_time} --------------"
done
end_time=`date +"%D %T"`
echo "---------------finish @ ${end_time} --------------"

set +x
