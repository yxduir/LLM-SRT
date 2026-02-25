data_config=yxdu/fleurs_eng_test_s2tt
hf download ${data_config} --repo-type dataset --local-dir ./fleurs_eng_test_s2tt
if [ -f "./fleurs_eng_test_s2tt/audio.tar.gz" ]; then
        echo "解压音频文件..."
        tar -zxvf "./fleurs_eng_test_s2tt/audio.tar.gz" -C "./fleurs_eng_test_s2tt/"
