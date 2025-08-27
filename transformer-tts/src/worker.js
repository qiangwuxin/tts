import {
  env, //配置AI模型运行环境 
  Tensor,//AI大模型处理数据的基本单位 表示高维数值数据的多维数组 
  AutoTokenizer, //AI 自行分词器 
  SpeechT5ForTextToSpeech, //文本转语音模型  得到语音的特征
  SpeechT5HifiGan, //语音合成模型  把语音特征和音色合成
} from '@xenova/transformers'
import {
  encodeWAV
} from './utils'
//huggingface 开源的大模型社区 
// 禁止使用本地模型,去请求远程的 tts 模型
env.allowLocalModels = false;
//transformer.js 文本->语音 tts 
//单例模式 核心难点 
//多次执行tts ai 业务，但是只会实例化一次
//他的实例化开销太大了，也没有必要 
class MyTextToSpeechPipeLine {
  //AI语音模型的数据源地址，用于下载不同说话人的声音特征向量
  //每个字，每个词
  static BASE_URL = 'https://huggingface.co/datasets/Xenova/cmu-arctic-xvectors-extracted/resolve/main/';
  //文本-> speech_tts 语音特征
  static model_id = 'Xenova/speecht5_tts'
  //语音特征->speecht5_hifigan->特有的角色音频文件
  static vocoder_id = 'Xenova/speecht5_hifigan'
  //分词器实例
  static tokenizer_instance = null;
  //模型实例
  static model_instance = null;
  //合成实例
  static vocoder_instance = null;
  static async getInstance(progress_callback = null) {
    //分词器实例化
    if (this.tokenizer_instance === null) {
      // 之前处理过的大模型，被预训练过的
      this.tokenizer_instance = AutoTokenizer.from_pretrained(this.model_id, {
        progress_callback
      })
    }
    if (this.model_instance === null) {
      //模型下载
      this.model_instance = SpeechT5ForTextToSpeech.from_pretrained(
        this.model_id,
        {
          dtype: 'fp32',
          progress_callback
        }
      )
    }
    if (this.vocoder_instance === null) {
      this.vocoder_instance = SpeechT5HifiGan.from_pretrained(
        this.vocoder_id,
        {
          dtype: 'fp32',
          progress_callback
        }
      )
    }
    return new Promise(async (resolve, reject) => {
      try {
        const result = await Promise.all([
          this.tokenizer_instance,
          this.model_instance,
          this.vocoder_instance
        ])
        self.postMessage({
          status: 'ready'
        });
        resolve(result)
      } catch (error) {
        reject(error)
      }
    })
  }
  static async getSpeakerEmbeddings(speaker_id) {
    try {
      const speaker_embeddings_url = `${this.BASE_URL}${speaker_id}.bin`;
      //张量
      //下载文件 .bin
      //转换数据 将.bin二进制数据转换为Float32Array
      //创建一个张量，构建一个1*512维度的特征向量
      const response = await fetch(speaker_embeddings_url);
      const arrayBuffer = await response.arrayBuffer();
      const speaker_embeddings = new Tensor(
        'float32',
        new Float32Array(arrayBuffer),
        [1, 512]
      );
      return speaker_embeddings;
    } catch (error) {
      console.error('Error fetching speaker embeddings:', error);
      throw error;
    }
  }
}
//es6 新增的数据结构 HashMap 先简单想象成JSON对象
const speaker_embeddings_cache = new Map();
self.onmessage = async (e) => {
  try {
    //ai pipeline 派发一个nlp任务
    //懒加载  llm初始化和加载放到第一次任务调用之时
    //解构三个实例
    const [tokenizer, model, vocoder] = await MyTextToSpeechPipeLine.getInstance(x => {
      self.postMessage(x)
    })

    const {
      input_ids
    } = await tokenizer(e.data.text);
    //token 将是LLM的输入
    // 将原始的输入，分词为一个一个word（最小单位），对应的数字编码（是bigint的）
    //向量的相似度、维度 了解万事万物了
    //一个一个token去生成
    //以前搜索的区别 
    //prompt->token->LLM(函数，向量计算，参数十亿+级别)->outpus

    //基于model生成的声音特征 
    // embeddings 向量计算
    let speaker_embeddings = speaker_embeddings_cache.get(e.data.speaker_id);
    if (speaker_embeddings === undefined) {
      // 下载某个音色的声音特征向量 
      speaker_embeddings = await MyTextToSpeechPipeLine.getSpeakerEmbeddings(e.data.speaker_id);
      //将下载的特征向量存入缓存
      speaker_embeddings_cache.set(e.data.speaker_id, speaker_embeddings)
    }

    const { waveform } = await model.generate_speech(
      input_ids,//分词数据
      speaker_embeddings,// 512 维的向量
      { vocoder } //合成器
    );

    // 声音的blob 文件
    const wav = encodeWAV(waveform.data);
    self.postMessage({
      status: 'complete',
      output: new Blob([wav], {
        type: 'audio/wav'
      })
    })
  } catch (error) {
    console.error('Error in worker:', error);
    self.postMessage({
      status: 'error',
      error: error.message
    });
  }
}