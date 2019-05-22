from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

# * Dữ liệu chỉ tải một lần rồi lưu lại để có thể sử dụng lần sau
dataset = fetch_20newsgroups()

# ! Phân tích dữ liệu
# * Các thuộc tính của dữ liệu
# print(dataset.keys())
# ? dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])

# * Danh sách các nhóm tin
# print(dataset['target_names'])
# ? ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 
# ? 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 
# ? 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 
# ? 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 
# ? 'talk.religion.misc']

# * Hiện nhóm tin dưới dạng số mà tin của tất cả các tin
# * Các nhóm tin được đánh số từ 0 đến 19
# print(dataset['target'])
# ? [7 4 4 ... 3 1 8]

# * Dữ liệu của set được lấy với thuộc tính data
# print(len(dataset.data))
# ? 11314

# * Xem thử một dữ liệu
# print(dataset.data[0])
# ? From: lerxst@wam.umd.edu (where's my thing)
# ? Subject: WHAT car is this!?
# ? Nntp-Posting-Host: rac3.wam.umd.edu
# ? Organization: University of Maryland, College Park
# ? Lines: 15
# ?
# ?  I was wondering if anyone out there could enlighten me on this car I saw
# ? the other day. It was a 2-door sports car, looked to be from the late 60s/
# ? early 70s. It was called a Bricklin. The doors were really small. In addition,
# ? the front bumper was separate from the rest of the body. This is 
# ? all I know. If anyone can tellme a model name, engine specs, years
# ? of production, where this car is made, history, or whatever info you
# ? have on this funky looking car, please e-mail.
# ?
# ? Thanks,
# ? - IL
# ?    ---- brought to you by your neighborhood Lerxst ----

# * Vẽ đồ thị với dữ liệu dataset
# * Số tin tức tương ứng với mỗi nhóm tin
# sns.distplot(dataset.target)
# plt.show()

# * tokenize các dữ liệu và tạo ra các vector với số lần xuất hiện của token
# * chỉ lấy 500 token xuất hiện nhiều nhất
# count_vector = CountVectorizer(max_features=500)
# data_count = count_vector.fit_transform(dataset.data)
# print(data_count[0].toarray())
# ? kết quả trả về là 1 sparse vector 500 chiều

# * có thể hiển thị danh sách các thuộc tính
# print(count_vector.get_feature_names())
# ? trả về là danh sách 500 token xuất hiện nhiều nhất trong cả bộ dữ liệu
# ? ['00', '000', '10', '100', '11', '12', '13', '14', '145', '15', '16', '17', '18', '19', '1993',
# ? '20', '21', '22', '23', '24', '25', '26', '27', '30', '32', '34', '40', '50', '93', 'a86', 'able',
# ? 'about', 'above', 'ac', 'access', 'actually', 'address', 'after', 'again', 'against', 'ago', 'all',
# ? 'already', 'also', 'always', 'am', 'american', 'an', 'and', 'andrew', 'another', 'answer', 'any',
# ? 'anyone', 'anything', 'apple', 'apr', 'april', 'are', 'armenian', 'around', 'article', 'as', 'ask',
# ? 'at', 'au', 'available', 'away', 'ax', 'b8f', 'back', 'bad', 'based', 'be', 'because', 'been',
# ? 'before', 'being', 'believe', 'best', 'better', 'between', 'bible', 'big', 'bill', 'bit', 'book',
# ? 'both', 'but', 'buy', 'by', 'ca', 'call', 'called', 'came', 'can', 'canada', 'cannot', 'car',
# ? 'card', 'case', 'cc', 'center', 'change', 'children', 'chip', 'christian', 'clipper', 'co', 'code',
# ? 'color', 'com', 'come', 'computer', 'control', 'could', 'course', 'cs', 'current', 'data', 'david',
# ? 'day', 'days', 'de', 'department', 'did', 'didn', 'different', 'disk', 'distribution', 'do', 'does',
# ? 'doesn', 'doing', 'don', 'done', 'dos', 'down', 'drive', 'during', 'each', 'edu', 'either', 'else',
# ? 'email', 'encryption', 'end', 'enough', 'etc', 'even', 'ever', 'every', 'everything', 'evidence',
# ? 'example', 'fact', 'far', 'fax', 'few', 'file', 'files', 'find', 'first', 'following', 'for', 'found',
# ? 'free', 'from', 'ftp', 'full', 'g9v', 'game', 'games', 'general', 'get', 'getting', 'give', 'given',
# ? 'go', 'god', 'going', 'good', 'got', 'gov', 'government', 'graphics', 'great', 'group', 'gun', 'had',
# ? 'hand', 'hard', 'has', 'have', 'having', 'he', 'heard', 'help', 'her', 'here', 'high', 'him', 'his',
# ? 'home', 'hope', 'host', 'how', 'however', 'hp', 'human', 'ibm', 'idea', 'if', 'image', 'important',
# ? 'in', 'inc', 'info', 'information', 'institute', 'interested', 'internet', 'into', 'is', 'isn',
# ? 'israel', 'issue', 'it', 'its', 'jesus', 'jews', 'jim', 'john', 'just', 'keep', 'key', 'keywords',
# ? 'kind', 'know', 'large', 'last', 'law', 'least', 'left', 'less', 'let', 'life', 'like', 'line',
# ? 'lines', 'list', 'little', 'live', 'll', 'local', 'long', 'look', 'looking', 'lot', 'mac', 'made',
# ? 'mail', 'make', 'makes', 'man', 'many', 'mark', 'max', 'may', 'maybe', 'me', 'mean', 'means',
# ? 'memory', 'message', 'michael', 'might', 'mike', 'mit', 'money', 'more', 'most', 'mr', 'much', 'must',
# ? 'my', 'name', 'nasa', 'national', 'need', 'net', 'netcom', 'never', 'new', 'news', 'next', 'nntp',
# ? 'no', 'non', 'not', 'note', 'nothing', 'now', 'number', 'of', 'off', 'old', 'on', 'once', 'one',
# ? 'only', 'open', 'opinions', 'or', 'order', 'org', 'organization', 'other', 'others', 'our', 'out',
# ? 'over', 'own', 'part', 'paul', 'pc', 'people', 'perhaps', 'person', 'phone', 'pl', 'place', 'play',
# ? 'please', 'point', 'possible', 'post', 'posting', 'power', 'president', 'pretty', 'price', 'probably',
# ? 'problem', 'problems', 'program', 'public', 'put', 'question', 'questions', 'quite', 'rather', 're',
# ? 'read', 'real', 'really', 'reason', 'remember', 'reply', 'research', 'right', 'rights', 'robert',
# ? 'run', 'running', 'said', 'sale', 'same', 'say', 'saying', 'says', 'science', 'scsi', 'second', 'see',
# ? 'seem', 'seems', 'seen', 'send', 'server', 'set', 'several', 'she', 'should', 'show', 'side', 'since',
# ? 'small', 'so', 'software', 'some', 'someone', 'something', 'source', 'space', 'speed', 'standard',
# ? 'start', 'state', 'states', 'steve', 'still', 'stuff', 'subject', 'such', 'sun', 'support', 'sure',
# ? 'system', 'systems', 'take', 'team', 'technology', 'tell', 'than', 'thanks', 'that', 'the', 'their',
# ? 'them', 'then', 'there', 'these', 'they', 'thing', 'things', 'think', 'this', 'those', 'though',
# ? 'thought', 'three', 'through', 'time', 'times', 'to', 'today', 'told', 'too', 'true', 'try',
# ? 'trying', 'turkish', 'two', 'type', 'uiuc', 'uk', 'under', 'university', 'unix', 'until', 'up', 'us',
# ? 'usa', 'use', 'used', 'using', 'uucp', 've', 'version', 'very', 'vs', 'want', 'war', 'was',
# ? 'washington', 'way', 'we', 'well', 'were', 'what', 'when', 'where', 'whether', 'which', 'while',
# ? 'who', 'whole', 'why', 'will', 'win', 'window', 'windows', 'with', 'without', 'won', 'word', 'work',
# ? 'works', 'world', 'would', 'writes', 'wrong', 'wrote', 'year', 'years', 'yes', 'yet', 'you', 'your']

# ! Tiền xử lý văn bản

# * Loại bỏ các số, từ có ký hiệu lạ như gạch ngang, dấu chấm,... chỉ dữ lại từ gồm các ký tự chữ
def is_letter_only(word):
    for i in word:
        if not i.isalpha():
            return False
    return True

data_clearned = []

for data in dataset.data:
    data_clearned.append(' '.join(word for word in data.split() if is_letter_only(word)))
    
# * tokenize kèm theo loại bỏ stop words với bộ english stop words scikit-learn
count_vector_sw = CountVectorizer(stop_words="english", max_features=500)
# data_count_cleaned_sw = count_vector_sw.fit_transform(data_clearned)
# print(count_vector_sw.get_feature_names())
# ? Danh sách các thuộc tính hiện tại đã đẹp hơn
# ? ['able', 'accept', 'access', 'according', 'act', 'actually', 'add', 'address', 'ago', 'agree', 'allow',
# ?  'american', 'anonymous', 'answer', 'anybody', 'apple', 'application', 'apr', 'april', 'area',
# ?  'argument', 'armenian', 'armenians', 'article', 'ask', 'asked', 'asking', 'assume', 'available',
# ?  'away', 'bad', 'based', 'basic', 'believe', 'best', 'better', 'bible', 'big', 'bike', 'bit', 'black',
# ?  'board', 'body', 'book', 'books', 'box', 'build', 'bus', 'buy', 'ca', 'california', 'called', 'came',
# ?  'canada', 'car', 'card', 'care', 'carry', 'case', 'cause', 'center', 'certain', 'certainly', 'change',
# ?  'check', 'children', 'chip', 'christian', 'christians', 'church', 'city', 'claim', 'clear', 'clinton',
# ?  'clipper', 'code', 'college', 'color', 'come', 'comes', 'coming', 'common', 'communications',
# ?  'company', 'computer', 'computing', 'consider', 'considered', 'contact', 'control', 'copy', 'correct',
# ?  'cost', 'couple', 'course', 'create', 'current', 'currently', 'cut', 'data', 'david', 'day', 'days',
# ?  'db', 'deal', 'death', 'department', 'did', 'difference', 'different', 'discussion', 'disk', 'display',
# ?  'division', 'dod', 'does', 'doing', 'dos', 'drive', 'driver', 'drivers', 'early', 'earth', 'easy',
# ?  'effect', 'email', 'encryption', 'end', 'engineering', 'especially', 'evidence', 'exactly', 'example',
# ?  'expect', 'experience', 'explain', 'fact', 'faith', 'far', 'fast', 'federal', 'feel', 'file', 'files',
# ?  'following', 'force', 'form', 'free', 'ftp', 'game', 'games', 'gave', 'general', 'gets', 'getting',
# ?  'given', 'gmt', 'god', 'goes', 'going', 'good', 'got', 'government', 'graphics', 'great', 'group',
# ?  'groups', 'guess', 'gun', 'guy', 'happened', 'hard', 'hardware', 'having', 'head', 'health', 'hear',
# ?  'heard', 'hell', 'help', 'high', 'history', 'hit', 'hockey', 'hold', 'home', 'hope', 'house', 'human',
# ?  'ibm', 'idea', 'image', 'important', 'include', 'includes', 'including', 'info', 'information',
# ?  'instead', 'institute', 'interested', 'interesting', 'international', 'internet', 'israel', 'israeli',
# ?  'issue', 'jesus', 'jewish', 'jews', 'john', 'just', 'key', 'keys', 'kill', 'killed', 'kind', 'know',
# ?  'known', 'knows', 'la', 'large', 'later', 'law', 'laws', 'league', 'leave', 'left', 'legal', 'let',
# ?  'level', 'life', 'light', 'like', 'likely', 'line', 'list', 'little', 'live', 'local', 'long',
# ?  'longer', 'look', 'looking', 'looks', 'lost', 'lot', 'love', 'low', 'mac', 'machine', 'mail', 'main',
# ?  'major', 'make', 'makes', 'making', 'man', 'mark', 'matter', 'maybe', 'mean', 'means', 'medical',
# ?  'members', 'memory', 'men', 'mentioned', 'message', 'michael', 'mike', 'military', 'million', 'mind',
# ?  'model', 'money', 'monitor', 'moral', 'na', 'nasa', 'national', 'near', 'need', 'needed', 'needs',
# ?  'network', 'new', 'news', 'nice', 'north', 'note', 'number', 'numbers', 'office', 'old', 'ones',
# ?  'open', 'opinions', 'order', 'original', 'package', 'particular', 'past', 'paul', 'pay', 'pc',
# ?  'people', 'period', 'person', 'personal', 'phone', 'place', 'play', 'players', 'plus', 'point',
# ?  'points', 'police', 'policy', 'political', 'position', 'possible', 'post', 'posted', 'posting',
# ?  'power', 'president', 'press', 'pretty', 'previous', 'price', 'private', 'probably', 'problem',
# ?  'problems', 'program', 'programs', 'project', 'protect', 'provide', 'public', 'question', 'questions',
# ?  'quite', 'radio', 'rate', 'read', 'reading', 'real', 'really', 'reason', 'recently', 'red',
# ?  'religious', 'remember', 'research', 'rest', 'return', 'right', 'rights', 'run', 'running', 'runs',
# ?  'said', 'sale', 'san', 'save', 'saw', 'say', 'saying', 'says', 'school', 'science', 'screen', 'scsi',
# ?  'second', 'section', 'security', 'seen', 'sell', 'send', 'sense', 'sent', 'serial', 'server',
# ?  'service', 'services', 'set', 'shall', 'short', 'similar', 'simple', 'simply', 'single', 'size',
# ?  'small', 'software', 'sort', 'sound', 'source', 'space', 'speak', 'special', 'specific', 'speed',
# ?  'standard', 'start', 'started', 'state', 'statement', 'states', 'steve', 'stop', 'strong', 'study',
# ?  'stuff', 'subject', 'sun', 'support', 'sure', 'systems', 'taken', 'takes', 'taking', 'talk', 'talking',
# ?  'team', 'technical', 'technology', 'tell', 'test', 'texas', 'text', 'thank', 'thanks', 'thing',
# ?  'things', 'think', 'thinking', 'thought', 'time', 'times', 'tin', 'today', 'told', 'took', 'total',
# ?  'tried', 'true', 'truth', 'try', 'trying', 'turkish', 'turn', 'type', 'understand', 'united',
# ?  'university', 'unix', 'unless', 'usa', 'use', 'used', 'user', 'uses', 'using', 'usually', 'value',
# ?  'various', 'version', 'video', 'view', 'vs', 'want', 'wanted', 'wants', 'war', 'water', 'way',
# ?  'went', 'western', 'white', 'willing', 'win', 'window', 'windows', 'women', 'word', 'work', 'working',
# ? 'works', 'world', 'worth', 'write', 'written', 'wrong', 'year', 'years', 'york', 'young']

# * Tổng hợp lại các bước tiền xử lý

data_clearned_v2 = []
# * Bộ dữ liệu tên riêng

all_names = set(names.words())

lemmatizer = WordNetLemmatizer()

for data in data_clearned:
    # * lemmatizing dữ liệu
    data_clearned_v2.append(' '.join(lemmatizer.lemmatize(word) for word in data.split() if word not in all_names))

data_clearned_v2_vector = count_vector_sw.fit_transform(data_clearned_v2)

# print(count_vector_sw.get_feature_names())
# ? Các thuộc tính bây giờ đã rất đẹp
# ? ['able', 'accept', 'access', 'according', 'act', 'action', 'actually', 'add', 'address', 'ago', 'agree',
# ?  'algorithm', 'allow', 'american', 'anonymous', 'answer', 'anybody', 'apple', 'application', 'apr',
# ?  'area', 'argument', 'armenian', 'armenians', 'article', 'ask', 'asked', 'assume', 'attack', 'attempt',
# ?  'available', 'away', 'bad', 'based', 'basic', 'belief', 'believe', 'best', 'better', 'bible', 'big',
# ?  'bike', 'bit', 'black', 'board', 'body', 'book', 'box', 'build', 'bus', 'business', 'buy', 'ca',
# ?  'california', 'called', 'came', 'car', 'card', 'care', 'carry', 'case', 'cause', 'center', 'certain',
# ?  'certainly', 'chance', 'change', 'check', 'child', 'chip', 'christians', 'church', 'city', 'claim',
# ?  'clear', 'clipper', 'code', 'college', 'color', 'come', 'coming', 'command', 'comment', 'common',
# ?  'communication', 'company', 'computer', 'computing', 'consider', 'considered', 'contact', 'control',
# ?  'copy', 'correct', 'cost', 'country', 'couple', 'course', 'court', 'cover', 'create', 'crime',
# ?  'current', 'cut', 'data', 'day', 'db', 'deal', 'death', 'department', 'device', 'did', 'difference',
# ?  'different', 'discussion', 'disk', 'display', 'division', 'dod', 'doe', 'does', 'doing', 'dos',
# ?  'drive', 'driver', 'drug', 'early', 'earth', 'easy', 'effect', 'email', 'encryption', 'end',
# ?  'engineering', 'entry', 'error', 'especially', 'event', 'evidence', 'exactly', 'example', 'expect',
# ?  'experience', 'explain', 'face', 'fact', 'far', 'fast', 'federal', 'feel', 'figure', 'file', 'final',
# ?  'following', 'food', 'force', 'form', 'free', 'friend', 'ftp', 'function', 'game', 'general',
# ?  'getting', 'given', 'gmt', 'god', 'going', 'good', 'got', 'government', 'graphic', 'great', 'ground',
# ?  'group', 'guess', 'gun', 'guy', 'ha', 'hand', 'hard', 'hardware', 'having', 'head', 'health', 'hear',
# ?  'heard', 'hell', 'help', 'high', 'history', 'hit', 'hockey', 'hold', 'home', 'hope', 'house', 'human',
# ?  'ibm', 'idea', 'image', 'important', 'include', 'includes', 'including', 'individual', 'info',
# ?  'information', 'instead', 'institute', 'interested', 'interesting', 'international', 'internet',
# ?  'israeli', 'issue', 'jewish', 'jews', 'job', 'just', 'key', 'kill', 'killed', 'kind', 'know', 'known',
# ?  'large', 'later', 'launch', 'law', 'le', 'lead', 'league', 'left', 'legal', 'let', 'level', 'life',
# ?  'light', 'like', 'likely', 'line', 'list', 'little', 'live', 'local', 'long', 'longer', 'look',
# ?  'looking', 'lost', 'lot', 'love', 'low', 'machine', 'mail', 'main', 'major', 'make', 'making', 'man',
# ?  'manager', 'matter', 'maybe', 'mean', 'medical', 'member', 'memory', 'men', 'message', 'method',
# ?  'military', 'million', 'mind', 'mode', 'model', 'money', 'monitor', 'month', 'moral', 'mouse', 'na',
# ?  'nasa', 'national', 'near', 'need', 'needed', 'network', 'new', 'news', 'nice', 'north', 'note',
# ?  'number', 'offer', 'office', 'old', 'open', 'opinion', 'order', 'original', 'output', 'package',
# ?  'particular', 'past', 'pay', 'pc', 'people', 'period', 'person', 'personal', 'phone', 'place',
# ?  'play', 'player', 'point', 'police', 'policy', 'political', 'position', 'possible', 'post', 'posted',
# ?  'posting', 'power', 'president', 'press', 'pretty', 'previous', 'price', 'private', 'probably',
# ?  'problem', 'product', 'program', 'project', 'provide', 'public', 'purpose', 'question', 'quite',
# ?  'radio', 'rate', 'read', 'reading', 'real', 'really', 'reason', 'recently', 'reference', 'religion',
# ?  'religious', 'remember', 'reply', 'report', 'research', 'response', 'rest', 'result', 'return',
# ?  'right', 'road', 'rule', 'run', 'running', 'said', 'sale', 'san', 'save', 'saw', 'say', 'saying',
# ?  'school', 'science', 'screen', 'scsi', 'second', 'section', 'security', 'seen', 'sell', 'send',
# ?  'sense', 'sent', 'serial', 'server', 'service', 'services', 'set', 'shall', 'short', 'shot',
# ?  'similar', 'simple', 'simply', 'single', 'site', 'situation', 'size', 'small', 'software', 'sort',
# ?  'sound', 'source', 'space', 'special', 'specific', 'speed', 'standard', 'start', 'started',
# ?  'state', 'statement', 'stop', 'strong', 'study', 'stuff', 'subject', 'sun', 'support', 'sure',
# ?  'systems', 'taken', 'taking', 'talk', 'talking', 'tape', 'tax', 'team', 'technical', 'technology',
# ?  'tell', 'term', 'test', 'texas', 'text', 'thanks', 'thing', 'think', 'thinking', 'thought', 'time',
# ?  'tin', 'today', 'told', 'took', 'total', 'tried', 'true', 'truth', 'try', 'trying', 'turkish', 'turn',
# ?  'type', 'understand', 'united', 'university', 'unix', 'unless', 'usa', 'use', 'used', 'user', 'using',
# ?  'usually', 'value', 'various', 'version', 'video', 'view', 'wa', 'want', 'wanted', 'war', 'water',
# ?  'way', 'weapon', 'week', 'went', 'western', 'white', 'widget', 'willing', 'win', 'window', 'windows',
# ?  'wish', 'woman', 'word', 'work', 'working', 'world', 'worth', 'write', 'written', 'wrong', 'year',
# ?  'york', 'young']

