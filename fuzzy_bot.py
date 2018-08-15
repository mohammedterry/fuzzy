class FuzzyBot:
    # this bot is robust to:
    #  spelling (using chargrams)
    #  phrasing (using question vector classification method)
    # and it can learn to speak from example conversations
    answers = {}
    answer_ids = {}
    word_vectors = {}
    chargram_vectors = {}

    def clean(self,txt):
        cleaned = ''
        for letter in txt.lower():
            if ord('a') <= ord(letter) <= ord('z') or letter.isspace():
                cleaned += letter
        return cleaned

    def chargrams(self,sentence,n=5):
        PAD = ' '*(n-1)
        chars = PAD + self.clean(sentence).lower() + PAD
        return {a+b+c+d+e for a,b,c,d,e in zip(chars,chars[1:],chars[2:],chars[3:],chars[4:])}

    def words(self,sentence):
        return set(self.clean(sentence).split())

    def learn(self, question, answer):
        # unsupervised learning of chargram vectors 
        a = answer.lower()

        # get answer id
        if a in self.answers:
            ans_id = self.answers[a]
        else:
            ans_id = len(self.answers)
            self.answer_ids[ans_id] = a
            self.answers[a] = ans_id

        # chargrams are used alongside words as they can be more robust to variations in spelling
        # update  vectors
        for word in self.words(question):
            if word not in self.word_vectors:
                self.word_vectors[word] = {ans_id}
            else:
                self.word_vectors[word] |= {ans_id}
        
        for chargram in self.chargrams(question):
            if chargram not in self.chargram_vectors:
                self.chargram_vectors[chargram] = {ans_id}
            else:
                self.chargram_vectors[chargram] |= {ans_id}
    
    def classify(self,utterance,weights = [1,1]):
        #use learnt word vectors to classify question 
        # each class corresponds to an answer
        # weights allow you to increase/decrease the relative contribution of words:chargrams
        word_weight,cg_weight = weights
        merged_vector = [0 for _ in range(len(self.answers))]

        for word in self.words(utterance):
            if word in self.word_vectors:
                vector = self.word_vectors[word]
                commonness = len(vector) #to weight rarer words more heavily
                for v in vector:
                    merged_vector[v] += word_weight/commonness 

        for chargram in self.chargrams(utterance):
            if chargram in self.chargram_vectors:
                vector = self.chargram_vectors[chargram]
                commonness = len(vector) 
                for v in vector:
                    merged_vector[v] += cg_weight/commonness 
        highest_value = max(merged_vector)
        if highest_value > 0:
            ans_ids = [i for i,count in enumerate(merged_vector) if count == highest_value]
            return [self.answer_ids[ans_id] for ans_id in ans_ids]   
        
    def batch_learn(self,questions_answers):
        # learn from a log of training examples 
        # input: list of tuples: [...(utterance,reply)...]
        for qa in questions_answers:
            self.learn(qa[0],qa[1])

    def chat(self,utterance):
        replies = self.classify(utterance)
        if replies is not None:
            if len(replies) > 1: 
                return '\n'.join(replies)
            return replies[0]
        return "???"

    def learn_live(self):
        # learn while talking to the user
        bot_reply = '???'
        while True:
            human_reply = input("\nuser: ")
            self.learn(bot_reply,human_reply)
            bot_reply = self.chat(human_reply)
            print('bot: {}'.format(bot_reply))



training_data = []
with open('training_examples.txt') as f:
    for line in f.readlines():
        q,a = line.split('§') 
        training_data.append((q.strip(),a.strip()))
b = FuzzyBot()
b.batch_learn(training_data)
print(b.classify("hi, how are you?"))
b.learn_live()
