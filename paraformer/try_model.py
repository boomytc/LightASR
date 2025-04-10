from funasr import AutoModel

# model = AutoModel(model="dengcunqin/speech_seaco_paraformer_large_asr_nat-zh-cantonese-en-16k-common-vocab11666-pytorch",
#                   model_revision="master"
#                   )
model = AutoModel(model="/home/ad/A/FunASR/model/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                  model_revision="master",
                  disable_update=True,
                  )

res1 = model.generate(input="/home/ad/视频/空地录音.wav",
                     hotword="",)
print(res1)

res2 = model.generate(input="/home/ad/视频/航班起降陆空通话流程示范.wav",
                     hotword="九幺三五 白云 放行",)
print(res2)
