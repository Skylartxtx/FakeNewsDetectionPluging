from flask import Flask, request, jsonify
import numpy as np
import paddle.fluid as fluid
import os
import math
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import logging
from pathlib import Path
import paddle
from flask import Flask
from flask_cors import CORS
import pandas as pd
import re
from wordcloud import WordCloud
import jieba
import jieba.posseg as pseg
import jieba.analyse 
import io
import base64
from collections import Counter
from snownlp import SnowNLP
from stanfordcorenlp import StanfordCoreNLP

# 初始化Flask应用（仅一次！）
app = Flask(__name__)
# 使用 FLASK_DEBUG 环境变量设置调试模式
flask_debug = os.getenv('FLASK_DEBUG', 'False')
app.debug = flask_debug.lower() == 'true'
# 允许所有域的CORS请求（开发阶段建议放宽限制）
CORS(app, resources={r"/*": {"origins": "*"}})
paddle.enable_static()
# 加载环境变量（确保.env文件在app.py同级目录）
env_path = Path(r"C:\Users\Skyla\Desktop\FakeNews\backend\.env")
load_dotenv(dotenv_path=env_path)

# 配置日志记录设置日志级别为 INFO。
logging.basicConfig(level=logging.INFO)
#创建一个日志记录器实例，用于记录日志信息。
logger = logging.getLogger(__name__)
# 数据库配置（从环境变量读取）
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),  # 添加默认值
    'port': os.getenv('DB_PORT', 3306),  # 带默认值
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER', 'detector_user'),       # 添加默认值
    'password': os.getenv('DB_PASSWORD'),
    'charset': 'utf8mb4',
    'auth_plugin': 'mysql_native_password',
    # 强制使用TCP连接（解决管道错误）
    'use_pure': True,
    'unix_socket': None
}
# 应用配置
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY'),
    MAX_CONTENT_LENGTH=int(os.getenv('MAX_TEXT_LENGTH', 2000))
)

# 模型路径配置
model_path = os.getenv('MODEL_PATH')
dict_path = os.getenv('DICT_PATH')

# 初始化PaddlePaddle
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
infer_exe.run(fluid.default_startup_program())
#连接数据库
def get_db_connection():
    """创建并返回数据库连接"""
    try:
        #mysql.connector.connect()：这是 mysql-connector-python 库提供的函数，用于建立与 MySQL 数据库的连接。
        conn = mysql.connector.connect(**DB_CONFIG)
        logger.info("成功连接到MySQL数据库")
        return conn
    except Error as e:
        logger.error(f"数据库连接失败: {e}")
        # 添加详细错误信息
        logger.error(f"当前配置: host={DB_CONFIG['host']}, port={DB_CONFIG['port']}, user={DB_CONFIG['user']}")
        raise


# 加载模型
model_path = "./backend/infer_model_Adam"
with fluid.program_guard(fluid.Program(), fluid.Program()):
    [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=model_path, executor=infer_exe)

AUTHORITATIVE_SOURCES = set()
try:
    df = pd.read_csv('backend/data/authoritative_sources.csv', header=None, encoding='gbk')
    AUTHORITATIVE_SOURCES = set(df[0].str.strip().unique())  # 去重并去除空格
    logger.info("权威新闻来源列表加载成功")
except Exception as e:
    logger.error(f"加载权威新闻来源失败: {str(e)}")
    exit(1)

# 加载字典
dict_path = "./backend/dict_Adam.txt"

try:
    with open(dict_path, 'r', encoding='utf-8') as f:
        dict_txt = eval(f.read())
    logger.info("字典加载成功")
except Exception as e:
    logger.error(f"字典加载失败：{str(e)}")
    exit(1)

def preprocess(text):
    """文本预处理函数"""
    data = []
    for s in text:
        if s not in dict_txt:
            s = '<unk>'
        data.append(dict_txt[s])
    return np.array(data, dtype=np.int64)


def check_source_authority(text):
    sources = []
    for source in AUTHORITATIVE_SOURCES:
        if source in text:
            sources.append(source)
    return ','.join(sources) if sources else '无来源'

# 加载中文停用词表（需准备停用词文件）
STOPWORDS = set()
try:
    with open('backend/data/stopwords.txt', 'r', encoding='utf-8') as f:
        STOPWORDS = set(line.strip() for line in f)
except Exception as e:
    logger.error(f"加载停用词失败: {str(e)}")
    exit(1)

#使用tfidf方法生成词云
def extract_keywords_tfidf(text, topK=100):
    """使用TF-IDF算法提取关键词"""
    keywords = jieba.analyse.extract_tags(
        text, 
        topK=topK, 
        withWeight=True, 
        allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'vn', 'v')  # 只保留名词和动词
    )
    return keywords

def generate_wordcloud(text):
    """
    生成词云图片并返回 Base64 编码
    
    参数:
    text: 待分析的文本
    auto_select: 是否根据文本长度自动选择提取方法
    use_tfidf: 是否强制使用TF-IDF算法提取关键词
    """
    # 分词并过滤停用词
    words = jieba.lcut(text)
    filtered_words = [word for word in words if word not in STOPWORDS and len(word) > 1]
    
    # 根据文本长度或参数选择关键词提取方法
    use_tfidf = len(filtered_words) > 100
    
    if use_tfidf:
        # 使用TF-IDF提取关键词和权重
        keywords = extract_keywords_tfidf(text)
        word_dict = {word: weight for word, weight in keywords}
    else:
        # 传统词频统计
        word_dict = {}
        for word in filtered_words:
            word_dict[word] = word_dict.get(word, 0) + 1
    
    # 创建词云对象
    wordcloud = WordCloud(
        font_path=r'C:\Users\Skyla\Desktop\FakeNews - 副本\backend\data\SanJiDianHeiJianTi-Zhong-2.ttf',
        width=800,
        height=400,
        background_color='white',
        stopwords=STOPWORDS,
        max_words=200,
        max_font_size=100,
        random_state=42,
        collocations=False  # 避免重复显示双词搭配
    ).generate_from_frequencies(word_dict)
    
    # 转为 Base64
    img_buffer = io.BytesIO()
    wordcloud.to_image().save(img_buffer, format='PNG')
    img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

def detect_ai(text):
    """基于统计特征的中文AI检测"""
    try:
        # 中文分词
        words = list(jieba.cut(text))
        
        # 过滤单字和标点（保留2字以上中文词汇）
        filtered_words = [word for word in words if len(word) > 1 and re.match(r'^[\u4e00-\u9fa5]+$', word)]
        
        # 计算词汇多样性
        unique_words = set(filtered_words)
        word_count = len(filtered_words)
        vocabulary_ratio = len(unique_words)/word_count if word_count > 0 else 0
        
        # 中文分句（按中文标点分割）
        sentences = re.split(r'[。！？；!?;]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        # 计算句子长度标准差（按字符数）
        sentence_lengths = [len(sent) for sent in sentences]
        sentence_length_std = np.std(sentence_lengths) if sentence_lengths else 0
        
        # 添加中文过渡词检测
        transition_words = {'首先', '其次', '此外', '最后', '总之', '同时', '然后', '接着', '而且', '并且', '况且', '何况', '再者', '另外', '反之', '否则', '但是', '然而', '不过', '却', '可是', '尽管', '虽然', '即使', '即便', '要是', '假如', '如果', '倘若', '一旦', '只要', '只有', '因为', '由于', '所以', '因此', '因而', '于是', '从而', '可见', '据此', '为此', '综上所述', '总而言之', '一言以蔽之', '简而言之', '具体来说', '举例来说', '比如', '例如', '诸如', '特别是', '尤其是', '与此同时', '与此相反', '与此类似', '除了','之外', '一方面','另一方面', '而且','不但', '还','不仅','不如', '宁可……也不', '尚且……何况','与其'}

        transition_count = sum(1 for word in words if word in transition_words)
        transition_ratio = transition_count / len(words) if len(words) > 0 else 0
        logger.info("transition_count:"+str(transition_count)+"  len(words):"+str(len(words)))
        
        # 调整后的判断逻辑（基于中文特征）
        is_AI = 1 if (
            vocabulary_ratio < 0.511 or      # 词汇重复率阈值
            sentence_length_std < 4 or      # 句子长度变化阈值
            transition_ratio >= 0.0200         # 过渡词使用频率阈值
        ) else 0
        logger.info("is_AI:"+str(is_AI)+"  vocabulary_ratio:"+str(vocabulary_ratio)+"  sentence_length_std:"+str(sentence_length_std)+"  transition_ratio:"+str(transition_ratio))
        return is_AI
        
    except Exception as e:
        logger.error(f"AI检测失败: {str(e)}")
        return 0
    
class ObjectivityAnalyzer:
    def __init__(self):
        # 加载主观词汇表
        self.subjective_words = set()
        self.load_subjective_words('backend/data/subjectwords_degrade.txt')
        self.load_subjective_words('backend/data/subjectwords_praise.txt')

        # 特殊句式关键词
        self.special_patterns = {
            'exclamation': [
                r'[！!]{2,}',  # 两个或更多感叹号
                r'太\w+了',
                r'真\w+啊',
                r'多么\w+啊',
                r'^[^。！？]*[！!]$'  # 以感叹号结尾的句子
            ],
            'question': [
                r'[?？]',
                r'(谁|什么|哪里|几时|多少|怎样|是不是|有没有|对不对)[?？]',
                r'难道\w+[?？]',
                r'怎么\w+呢[?？]',
                r'何不\w*[?？]',
                r'为何\w*[?？]'
            ],
            'imperative': [
                r'(请|务必|必须|要|赶快|赶紧|马上|禁止|切勿)\w*',
                r'(别|不要|千万别)\w*',
                r'^\w+(吧|呀)$'
            ],
            'parallel': [r'(\w+)(，|;)\1', r'(\w+)又\1', r'(\w+)也\1'],
            'emphasis': [r'确实\w+', r'真的\w+', r'绝对\w+', r'千万\w+']
        }

    def load_subjective_words(self, file_path):
        """从指定文件加载主观词汇"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.subjective_words.add(line.strip())
        except FileNotFoundError:
            logger.error(f"文件 {file_path} 未找到。")

    def detect_special_sentences(self, text):
        """检测特殊句式"""
        # 先分句 - 改进分句逻辑，正确处理连续标点
        sentences = re.split(r'[。！？;]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        total_sentences = len(sentences)

        special_counts = {
            'exclamation': 0,
            'question': 0,
            'imperative': 0,
            'parallel': 0,
            'emphasis': 0
        }
        total_special = 0

        # 预处理：合并连续的感叹号为一个
        processed_text = re.sub(r'[！!]+', '！', text)

        for sentence in sentences:
            # 检测感叹句
            is_special = False
            for pattern in self.special_patterns['exclamation']:
                if re.search(pattern, sentence):
                    special_counts['exclamation'] += 1
                    is_special = True
                    break

            # 检测疑问句
            for pattern in self.special_patterns['question']:
                if re.search(pattern, sentence):
                    special_counts['question'] += 1
                    is_special = True
                    break

            # 检测祈使句
            for pattern in self.special_patterns['imperative']:
                if re.search(pattern, sentence):
                    special_counts['imperative'] += 1
                    is_special = True
                    break

            # 检测排比句
            for pattern in self.special_patterns['parallel']:
                if re.search(pattern, sentence):
                    special_counts['parallel'] += 1
                    is_special = True
                    break

            # 检测强调句
            for pattern in self.special_patterns['emphasis']:
                if re.search(pattern, sentence):
                    special_counts['emphasis'] += 1
                    is_special = True
                    break
            if is_special:
                total_special += 1

        # 额外检查：确保感叹号结尾的句子被正确识别
        # 直接检查原始文本中的感叹句数量
        exclamation_sents = re.findall(r'[^！!]*[！!]+', text)
        special_counts['exclamation'] = len([s for s in exclamation_sents if s.strip()])
        total_special = special_counts['parallel']+special_counts['exclamation'] +special_counts['question']+special_counts['imperative']+special_counts['emphasis']
        if total_special>=total_sentences and total_sentences !=0:
            special_ratio=1
        else:
            special_ratio=total_special / total_sentences if total_sentences > 0 else 0

        return {
            'total_sentences': total_sentences,
            'special_counts': special_counts,
            'total_special': total_special,
            'special_ratio': special_ratio ,
            'sentences': sentences  # 返回分句结果用于调试
        }

    def analyze(self, text):
        """综合客观性分析"""
        # 情感倾向分析
        sentiment_score = SnowNLP(text).sentiments

        # 主观词汇检测
        words = pseg.cut(text)
        subj_word_count = 0
        total_words = 0
        for word, flag in words:
            if flag in ['a', 'ad', 'ag', 'an'] and word in self.subjective_words:  # 形容词/副词类
                subj_word_count += 1

            total_words += 1

        # 特殊句式分析
        special_sentence_info = self.detect_special_sentences(text)
        special_sentence_ratio = special_sentence_info['special_ratio']

        # 计算各项指标
        subj_ratio = subj_word_count / total_words if total_words > 0 else 0
        logger.info("情感模型得分:" + str(sentiment_score))
        emotion_intensity = abs(0.5 - sentiment_score) * 2  # 0-1
        logger.info(f"主观词汇率:{subj_ratio}  情感强度分析:{emotion_intensity}  特殊句式率:{special_sentence_ratio}  ")
        logger.info(f"特殊句式详情: 感叹句:{special_sentence_info['special_counts']['exclamation']} "
                    f"疑问句:{special_sentence_info['special_counts']['question']} "
                    f"祈使句:{special_sentence_info['special_counts']['imperative']} "
                    f"排比句:{special_sentence_info['special_counts']['parallel']} "
                    f"强调句:{special_sentence_info['special_counts']['emphasis']}")
        logger.info(f"总句子个数: {special_sentence_info['total_sentences']}")
        logger.info(f"分句结果: {special_sentence_info['sentences']}")  # 输出分句结果用于调试

        # 综合评分（可根据需求调整权重）
        score = 100 - (
                emotion_intensity * 20+  # 情感强度占比20%
                math.pow(subj_ratio, 1 / 3) * 45 +  # 主观词占比45%
                special_sentence_ratio * 35  # 特殊句式占比30%
        )

        return round(score, 2)

    
# 初始化客观性分析器
objectivity_analyzer = ObjectivityAnalyzer()

#保存到数据库函数
def save_to_database(text, is_fake, probability,Media_sources,wordcloud,is_AI,objectivity_score):
    """保存记录到MySQL数据库"""
    conn = None
    #验证词云的最大长度
    max_length = 4294967295
    if len(wordcloud) > max_length:
        logger.error("词云数据过长，无法插入数据库")
        return
    #尝试建立数据库连接
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO detection_records 
            (content, is_fake, probability,Media_sources,wordcloud,is_AI,objectivity_score)
        VALUES 
            (%s, %s, %s, %s,%s,%s,%s)
        """
        cursor.execute(insert_query, (text, is_fake, probability,Media_sources,wordcloud,is_AI,objectivity_score))
        conn.commit()
        logger.info(f"成功保存记录：{text[:50]}...")
        
    except Error as e:
        logger.error(f"数据库操作失败: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

#    定义了一个 Flask 路由 /detect，只接受 POST 请求。
#   当客户端发送 POST 请求到 /detect 时，会调用 detect 函数。
@app.route('/detect', methods=['POST'])
def detect():
    """处理检测请求的主端点"""
    try:
        # 验证输入
        if 'text' not in request.json or not request.json['text'].strip():
            return jsonify({'error': '请输入有效的新闻内容'}), 400
            
        text = request.json['text'].strip()
###权威性检查
        Media_sources = check_source_authority(text)
##判断是否为AI
        is_AI = detect_ai(text)
### 生成词云
        wordcloud = generate_wordcloud(text)
### 新增客观性分析
        objectivity_score = objectivity_analyzer.analyze(text)
###使用模型进行推理       
        # 文本预处理
        processed_data = preprocess(text)
        
        # 创建一个 LoD Tensor（层次化张量），用于将预处理后的数据输入到 PaddlePaddle 模型中。
        tensor_words = fluid.create_lod_tensor(
            [processed_data], 
            [[len(processed_data)]], 
            place
        )
        
        # 推理
        with fluid.program_guard(infer_program, fluid.Program()):
            # 推理
            result = infer_exe.run(
                program=infer_program,
                feed={feeded_var_names[0]: tensor_words},
                fetch_list=target_var
            )

#### 解析结果
        
        prob_fake = float(result[0][0][-1])
        if(prob_fake < 0.5):
            is_fake = 1
        else:
            is_fake = 0
        if(prob_fake<0.5):
            probability = 100-round(prob_fake * 100, 2)
        else:
            probability = round(prob_fake * 100, 2)

               
        # 保存到数据库
        save_to_database(text, is_fake, probability,Media_sources,wordcloud,is_AI,objectivity_score)
        return jsonify({
            'text': text,
            'is_fake': is_fake,
            'probability': probability,
            'Media_sources': Media_sources,  
            'wordcloud': wordcloud, 
            'is_AI': is_AI, 
            'objectivity_score': objectivity_score,  
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"请求处理失败：{str(e)}")
        return jsonify({
            'status': 'error',
            'message': '服务器内部错误'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # 调试模式输出更详细错误