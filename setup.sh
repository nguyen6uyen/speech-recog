mkdir -p benchmarks/LRS3/language_models/lm_en_subword/
# Download Language Model (lm_en_subword)
python3 -m huggingface_hub.cli download Amanvir/lm_en_subword model.json --local-dir benchmarks/LRS3/language_models/lm_en_subword --local-dir-use-symlinks False
python3 -m huggingface_hub.cli download Amanvir/lm_en_subword model.pth --local-dir benchmarks/LRS3/language_models/lm_en_subword --local-dir-use-symlinks False

mkdir -p benchmarks/LRS3/models/LRS3_V_WER19.1/
# Download VSR Model (LRS3_V_WER19.1)
python3 -m huggingface_hub.cli download Amanvir/LRS3_V_WER19.1 model.json --local-dir benchmarks/LRS3/models/LRS3_V_WER19.1 --local-dir-use-symlinks False
python3 -m huggingface_hub.cli download Amanvir/LRS3_V_WER19.1 model.pth --local-dir benchmarks/LRS3/models/LRS3_V_WER19.1 --local-dir-use-symlinks False