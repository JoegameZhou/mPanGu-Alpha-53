# UTF-8

import sentencepiece as spm
import jieba

# langs_ID = {'zh': 128301, 'ko': 128302, 'vi': 128303,
#             'bg': 128304, 'ru': 128305, 'sr': 128306, 'uk': 128307,
#             'ar': 128308, 'ps': 128309, 'fa': 128310, 'he': 128311,
#             'bn': 128312, 'ur': 128313, 'ne': 128114, 'hi': 128315, 'ta': 128316,
#             'de': 128317, 'en': 128318,
#             'es': 128102, 'fr': 128100, 'pt': 128101,
#             'mn': 128103, }
# translate_ID = 128300

langs_ID = {'zh': 128301, 'ko': 128302, 'vi': 128303,
            'de': 128317, 'en': 128318, 'nl': 128132,
            'ms': 128109, 'id': 128110, 'tl': 128111,
            'mn': 128103, 'my': 128104, 'th': 128105, 'lo': 128106, 'km':128107,
            'lt': 128112, 'et': 128113, 'lv': 128133, 'hu': 128115,
            'pl': 128116, 'cs': 128117, 'sk': 128118, 'sl': 128119, 'hr': 128120, 'bs': 128121, 'sr': 128306, 'bg': 128304,
            'mk': 128122, 'ru': 128305, 'uk': 128307, 'be': 128123,
            'sq': 128124, 'el': 128125, 'ka': 128126, 'hy': 128127,
            'ro': 128108, 'fr': 128100, 'es': 128102, 'pt': 128101,
            'fa': 128310, 'he': 128311, 'ar': 128308, 'ps': 128309,
            'tr': 128128, 'kk': 128129, 'uz': 128130, 'az': 128131,
            'hi': 128315, 'ta': 128316, 'ur': 128313, 'bn': 128312, 'si': 128314, 'ne': 128114}

translate_ID = 128300



class SpmTokenizer(object):

    def __init__(self, model_file):

        self.sp = spm.SentencePieceProcessor(model_file=model_file)

        self.specialIDNum = 300
        self.eod_id = self.vocab_size - 1
        self.eot_id = self.vocab_size - 2
        self.pad_id = self.vocab_size - 3

        ### pad id : 128297
        ### eot id : 128298
        ### vocab size : 128300
        ### 128320

    @property
    def vocab_size(self):
        return self.sp.vocab_size() + self.specialIDNum

    @property
    def spmVocabSize(self):
        return self.sp.vocab_size()

    @property
    def eod(self):
        return self.eod_id

    def tokenize(self, text):
        """ Tokenize a string. """
        return self.sp.encode(text)

    def convert_tokens_to_ids(self, tokens):
        return tokens

    def convert_ids_to_tokens(self, ids):
        ids = [id if id < self.sp.vocab_size() else 0 for id in ids]
        return self.decode(ids)

    def encode(self, text):
        res = self.tokenize(text)
        return res

    def decode(self, tokens):
        text = self.sp.decode(tokens)
        text = text.replace('\u2583', '\n')
        return text


if __name__ == '__main__':
    # Hindi_text = 'उत्साह बढ़ाने वाली कविताप्रेरणा दायक कविताप्रेरणा देने वाली कविताप्रेरणादायक कविता बच्चों के लिएप्रेरणादायक गजलप्रेरणादायक शायरी इन हिंदीप्रेरणादायक सुविचारप्रेरणादायक हिन्दी कविताप्रेरणादायी कविता हिंदीमनोबल बढ़ाने वाली कवितामनोबल बढ़ाने वाले विचारसकारात्मक कवितायेसकारात्मक सुविचारसकारात्मक सोच पर कविताहौसला पर कविताहौसला पर शायरीहौसला बढ़ाने वाली कविताहौसला बढ़ाने वाले विचारहौसला बढ़ाने वाले सुविचारज़िंदादिल ज़िन्दगी कविताज़िन्दगी पर कविता'
    # Chinese_text = '湖人 板\n凳双枪乔丹-法玛尔和萨沙-武贾西奇回访斯台普斯，面对旧东家他们肯定想有所表现。第二节的故事是香农-布朗PK武贾西奇，奥多姆PK汉弗里斯，都是有故事的人。汉弗里斯篮下自投自抢拿下2分，别看情场对付卡戴珊的本领不如人，可到了球场上可不输奥多姆。小布朗单挑萨沙得手一次就没了声息，但湖人的整体进攻却哑火，篮网打的没多少章法，但依然扩大了领先优势，奥多姆没人防守的快攻上篮都不进，湖人问题不小。'
    # Urdu_text = 'حماسه‌آفرینی جدید بسیجیان سلحشور در برگزاری رزمایش سراسری اقتدار عاشورایی بسیج تحت عنوان سپاهیان حضرت محمد رسول الله (ص) که یادآور اعزام غرورآفرین سپاهیان حضرت محمد رسول‌الله در دوران دفاع مقدس است، برگ زرین دیگری در کارنامه درخشان بسیج است که با حضور ده‌ها هزار نیرو بسیجی فداکار و داوطلب در عرصه‌های مختلف دفاعی، امنیتی، خدمات‌رسانی،‌محرومیت‌زدایی، بهداشتی و درمانی و غیره جلوه‌ای از همدلی، ایمان، قدرت و عمل انقلابی را به نمایش گذاشت.'
    # Thai_text = 'ใน VDO ชุดนี้นะครับเราจะมาพูดถึง Amibroker Backtest นะ Backtest คืออะไร Backtest คือการทดสอบระบบของเรานะครับ ระบบซื้อ-ขาย ว่าระบบซื้อ-ขายของเราเนี่ยทำงานได้ดีขนาดไหน ทำกำไรให้เราขนาดไหน หรือว่าทำให้เราเจ๊งขนาดไหนนะครับ เพราะฉะนั้นเนี่ยเราจะมาดูในส่วนนี้กัน คราวนี้เดี๋ยวเปิด Amibroker ขึ้นมาก่อนนะครับโดยที่ในส่วนของ Backtest เนี่ย หลักๆมีส่วนประกอบอะไรบ้าง มาที่ปุ่มบวกนี่นะครับเพิ่ม windows นะครับ เห็นมั้ยครับแล้วเลื่อนลงมาในส่วนของ Backtest เนี่ยจะประกอบด้วย 2 ส่วนหลักๆคือส่วนของ Analysis document นี่นะครับเป็นการตั้งค่าว่าจะ test จากวันไหนถึงวันไหน จะ test หุ้นอะไรบ้าง ลองเปิดขึ้นมาดูแล้วกันนี่นะครับ Analysis ก็จะเห็นว่ามี Backtest อยู่ตรงนี้แล้วก็จะ test ตามอะไร ตามสูตรในนี้ อันนี้ก็เป็นสูตรที่ผมเขียนค้างไว้นะครับอันนี้ไม่เป็นไร เราก็จะ test กับหุ้นอะไรบ้างเนี่ย ก็บอกว่า test หุ้นทั้งหมดนะครับโดย test ในช่วงไหนบ้างนี่นะครับก็จะมีให้ set ได้ว่า test วันล่าสุดหรือ test bar ล่าสุดนี่นะครับ อันนี้คือ test จากวันไหน from ถึง to นะครับแล้วก็ตั้งค่าได้ในนี้ว่าจะเอาเป็นวันที่เท่าไร หรือพิมพ์ก็ได้นี่นะครับ พิมพ์เข้าไปของผมนี่เป็น format ของอังกฤษนะครับก็คือจะเป็น เดือน/วัน/ปีถ้าท่านใช้ format เป็นไทย จะขึ้นเป็น วัน/เดือน/ปี นะครับแล้วแต่ว่า windows คิดยังไงอันนี้เข้าใจตรงกันนะครับ คราวนี้สมมุติผมพิมพ์เข้าไปผมเปลี่ยนเป็น2012 เห็นมั้ยครับ ก็เปลี่ยนเดือนของผมตรงนี้แต่เดือนของท่านอาจอยู่ตรงนี้ก็ได้นะครับแล้วแต่ format ผมเปลี่ยนกลับแล้วกัน 2011 อันนี้ก็เป็นส่วนของ Analysis ต่อไปถ้าจะต้องมีส่วนของ windows ด้วย อ้าวส่วนของ formula ด้วยว่าท่านจะเขียนเงื่อนไขในการซื้อ ขายอย่างไรก็กด บวกนะครับแล้วก็บอก new formula คราวนี้จะมี windows โผล่ขึ้นมา มาโผล่อีกหน้าต่างนึง อ่ะอันนี้ก็เป็น windows ของผมเห็นมั้ยครับก็เป็น windows ที่เอาไว้เขียน formula โดยที่ formula ที่ท่านเขียนขึ้นมาเองเนี่ยมันจะไม่ไปรวมกับ default ของ Amibroker อันนี้เป็น default นะครับ เดี๋ยวปิดเข้าไปส่วน code formula ที่ผมเขียนขึ้นมาเนี่ย มันจะไปอยู่ในส่วนของ customs นี่นะครับ ผมก็มีเขียนอะไรทิ้งไว้นะครับก็ว่ากันไป อันนี้ก็ให้ทำความเข้าใจนะครับว่าในส่วนของ backtest เนี่ยประกอบด้วย 2 ส่วนหลักๆก็คือ new Analysis … Analysis document หรือ Formula 2 ส่วนนี้นะครับเดี๋ยวเราจะมาพูดถึง 2 ส่วนนี้กัน ว่าไปทีละส่วน ทีนี้ในส่วนของ backtest เนี่ยเป็นส่วนที่ยากที่สุดไม่ว่าจะเป็น level ของ introduction เป็น level basic advance หรือ inter media ก็ตามเพราะงั้นจะใช้เวลาในส่วนนี้เยอะสุด Ok นะครับ'
    # Malay_text = 'ചായ കുടിയും പത്രം വായനയും \n കഴിഞ്ഞ ഹൈദ്രോസിക്കായുടെ ഒപ്പം സുലമാനിക്കയും എഴുന്നേറ്റു. പതുക്കെ കൂടെ നടന്ന് അമേരിക്കന്‍ വിശേഷങ്ങള്‍ എടുത്തിട്ടങ്ങലക്കി. അതോടെ ഹൈദ്രോസിക്ക കോടീശ്വരനില്‍ ചോദ്യം ചോദിക്കുന്ന സുരേഷ് ഗോപിയുടെ വീറും വാശിയോടും കൂടെ ചോദ്യങ്ങള്‍ ഓപഷനോടു കൂടിയും ഇല്ലാതെയും വീശി തുടങ്ങി. മണിച്ചിത്രത്താഴില്‍ നാഗവല്ലിയെ ഡാന്‍സ് ചെയ്ത് കൊണ്ടു പോകുന്നത് പോലെ ചോദ്യങ്ങള്‍ക്കുത്തരങ്ങളും കൊടുത്ത് കഥകളും പറഞ്ഞ് സുലൈമാനിക്ക മിഷന്‍ സക്സസ് ആക്കി. പാടവരമ്പിലെത്തി പാടമെല്ലാം കണ്ടിട്ടും ഹൈദ്രോസിക്കാന്റെ കരിനാക്കില്‍ നിന്നൊന്നും വരുന്നില്ല. അവസാനം സുലൈമാനിക്ക തന്നെ വിഷയം എടുത്തിട്ടു.'
    # Arabic_text = 'با توجه به آنچه گفته شد، میدان مطالعاتی مرزها در ایران با محدودیت‌های اندیشگی، رشته‌ای و نهادی گسترده‌ای همراه بوده است. بیشتر این مطالعات محصور در حوزه‌ی جغرافیای سیاسی، انتظامی، بین‌المللی و سیاسیِ صرف بوده است. این در حالی است که مرزهای سیاسی در ایران، به‌لحاظ فرهنگی مرزهای گشوده‌ای هستند که نشان از فضای گسترده‌ی ایران فرهنگی دارند. بر این اساس، مرز بیشتر از آن‌که به‌معنای امتناع باشد، فضایی حیاتی و زیسته‌ی زیست-جهان‌ها و فرهنگ‌هایی است که حیات و چالش‌های آن‌ها موضوعاتی ملی است.'

    tokenizer = SpmTokenizer('spm.128k.model.1')
    # tokens = tokenizer.tokenize(Chinese_text)
    # ids = tokenizer.convert_tokens_to_ids(tokens)
    # txt = tokenizer.convert_ids_to_tokens(ids)

    line1 = '34'
    line2 = '4434'
    a = f"{line1} _☪☣_ {'zh'}-{'ar'} _☪☣_ {line2}"
    b = tokenizer.tokenize(a)
    aa  = '使 妇女 更 容易 感染 艾滋病毒 的 原因 还 包括 受 教育 机会 不 平等 ， 其中 包括 全面 的 性 教育 和 艾滋病毒 防治 教育 ， 很难 获得 固定收入 和 就业 ， 缺乏 经济 安全 ， 以及 暴力 和 恐惧 。'
    tokens2 = tokenizer.tokenize(aa)
    tokens2 = [i for i in tokens2 if i != 119132]
    tokens3 = tokenizer.tokenize(''.join(aa.split()))
    tokens3 = [i for i in tokens3 if i != 119132]
    for i in tokens2:
        if i != 119132:
            print(tokenizer.convert_ids_to_tokens([i]))
    for i in tokens3:
        print(tokenizer.convert_ids_to_tokens([i]))
    aaa = ' '.join(jieba.cut(''.join(aa.split()).strip()))
    print(txt)

    pass





