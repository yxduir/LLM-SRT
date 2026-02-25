data_config=yxdu/multi30k_test_tts_smt
hf download ${data_config} --repo-type dataset --local-dir ./multi30k_test_tts_smt
if [ -f "./multi30k_test_tts_smt/audio.tar.gz" ]; then
        echo "解压音频文件..."
        tar -zxvf "./multi30k_test_tts_smt/audio.tar.gz" -C "./multi30k_test_tts_smt/"