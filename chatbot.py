#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 16:59:14 2018

@author: richard
"""

import numpy as np
import tensorflow as tf
import re #print the text and replcaed some characters, to simply the conversation
import time


#importing the dataset

lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n') # to read the dataset
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n') # to read the dataset

#making dictionary, maps each line with the title.
#input, outpunt. 
#Dictionary = {'ID of its line': the line itself}
id2line = {}
for line in lines:
    #splite it by the +++$++++
    _line = line.split(' +++$+++ ')
    if len(_line) ==5 :
        id2line[_line[0]] = _line[4] #Create the dictionar
        
#Creating the list of conversation
conversations_ids = []
for conversation in conversations[:-1]: # the last row is empty 
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ","") #keep the comma, we need it to perform the splite again
    conversations_ids.append(_conversation.split(","))
    
#Getting sepereately the quesitons and answers
#two huge lists, the same size of them
# Question[i] ==> Answer[i] # the input and output
questions = []
answers = []
for conversation in conversations_ids:
    for L in range(len(conversation) - 1): 
        questions.append(id2line[conversation[L]])
        answers.append(id2line[conversation[L+1]])

#Doing the first cleaning of the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"\'ll"," will",text)
    text = re.sub(r"\'ve"," have",text)
    text = re.sub(r"\'re"," are",text)
    text = re.sub(r"\'d"," would",text)
    text = re.sub(r"won't","will not",text)
    text = re.sub(r"can't","can not",text)
    text = re.sub(r"don't","do not",text)
    text = re.sub(r"[-(),\"#/@;:<>{}+,?.|=]", "",text)
    return text

#Cleaning the quesiton 
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

#why we can't use #clean_question = map(lambda x: clean_text(x),questions)

#Cleaninig the answer

clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

#creating a dictionary that maps each word to its number of occurance.
word2count={}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1 
        
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1         

#creating two dictionaries that map the question word and answer word to unique integer.

threshold = 20 # delete 5 %

questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1



answerswords2int = {}
word_number = 0
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1
        
#adding the last tokens to these two dictionary
#used for encoding or decoding
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int)+1
for token in tokens:
    answerswords2int[token] = len(answerswords2int)+1

#create the inverse dictionary of the answerswords2int dictionary
#cuz we need the inverse mapping in the end of decoding
answersint2word = {w_i: w for w, w_i in answerswords2int.items()} #inverse

#Add the <EOS> to the end of every string 
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>' # see <EOS> as a word

#tralasting all the questions and answers into int
#replace all the words filted out by <OUT>
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)


answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

# sorting quesitons and answers by the length of questions
# speed up the traninig and reduce the loss

sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1,25 + 1):
    #get indext of question, and the quesition
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
            
        



###################### PART TWO ######################
            
# Creating placeholders for the inputs and the target
            
def model_inputs():
    #convert the input to tensorflow placeholder
    inputs = tf.placeholder(tf.int32, [None,None], name='input') # create the placeholder, (1) type (2)Dimension
    targets = tf.placeholder(tf.int32, [None,None], name='targets')
    #two more tenserflow placeholder 1.learninig rate 2.dropout rate
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, lr, keep_prob

#preprocessing the targets
#the decoder will only accept the certain format of targets
#the target has to be batches, which means we have to chose the batch size 
#adding <SOS> token
    
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size,1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets,[0,0], [batch_size, -1],[1,1]) # take all the lines and the columns except last one
    preprocessed_targets = tf.concat([left_side,right_side], axis = 1) # axis: Horizontal
    return preprocessed_targets
#Creating the Encoder RNN Layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length): #main layer
    #creating LSTM
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    #apply the dropout to the LSTM
    lstm_dropoupt = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob= keep_prob) # keep_prob to control the dropout rate
    #creating the encoder cell
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropoupt]*num_layers)
    # the input size of the forward cell and backward cell must match
    #only need encoder_state
    encoder_output,encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw= encoder_cell, #it's a output, forward
                                                      cell_bw= encoder_cell, #on 2 direction but on the same encoder_cell
                                                      sequence_length = sequence_length,
                                                      inputs = rnn_inputs,
                                                      dtype= tf.float32
                                                      ) 
    return encoder_state

#decoder, cross-validation 
#step: trainig
    
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size): #decodeing scope is a object of tf.variable scope
    #get the attention state
    attention_state = tf.zeros([batch_size,1, decoder_cell.output_size]) #initialise 
    #keys --> target states, attention_values --> the values used to construct the context vector, 
    #attention_score_function --> compute the similarity between the keys and target states
    #attention_construct_function --> used to build attention state
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state, attention_option='bahdanau', num_units= decoder_cell.output_size)
    #before this function, the attention has to be prepared first
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], 
                                                                              attention_keys,   
                                                                              attention_values, 
                                                                              attention_score_function, 
                                                                              attention_construct_function,
                                                                              # a name scope for decoder function 
                                                                              name = 'attn_dec_train'
                                                                              )
    #only need decoder_output
    decoder_output,decoder_final_state,decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,   
                                                                                                            training_decoder_function,  
                                                                                                            decoder_embedded_input, 
                                                                                                            sequence_length,
                                                                                                            scope= decoding_scope
                                                                                                            )
    #apply the final drop out
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

#Decoding the test/validation set
#new observation that won't be used on the traininig
#reduce overfitting and improve the accuracy

def decode_test_set(encoder_state, decoder_cell, decoder_embedding_metrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size): #decodeing scope is a object of tf.variable scope
    #get the attention state
    attention_state = tf.zeros([batch_size,1, decoder_cell.output_size]) #initialise 
    #keys --> target states, attention_values --> the values used to construct the context vector, 
    #attention_score_function --> compute the similarity between the keys and target states
    #attention_construct_function --> used to build attention state
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state, attention_option='bahdanau', num_units= decoder_cell.output_size)
    #before this function, the attention has to be prepared first
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function, 
                                                                              encoder_state[0], 
                                                                              attention_keys,   
                                                                              attention_values, 
                                                                              attention_score_function, 
                                                                              attention_construct_function,
                                                                              # a name scope for decoder function 
                                                                              decoder_embedding_metrix, 
                                                                              sos_id,   
                                                                              eos_id,   
                                                                              maximum_length,   
                                                                              num_words,
                                                                              name = 'attn_dec_inf'
                                                                              )
    #only need decoder_output
    test_prediction, decoder_final_state,decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,   
                                                                                                              test_decoder_function,  
                                                                                                              #decoder_embedded_input,  #only used for training 
                                                                                                              #sequence_length,
                                                                                                              scope= decoding_scope
                                                                                                              )
    return test_prediction

#creating decoder RNN                                            #encoder_state--> the output of encoder,input of decoder 
def decoder_rnn(decoder_embedded_input, decoder_embedding_metrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("docoding") as decoding_scope:
        #create the LSTM layer
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        #apply dropout regulisation
        lstm_dropoupt = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob= keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropoupt]*num_layers)
        #initialise the wieght
        weights= tf.truncated_normal_initializer(stddev=0.1)
        #initialise the bias
        biases = tf.zeros_initializer()
        #The weight and bias for fully connected layer<the end of RNN>
        #make the output function
        output_function = lambda x : tf.contrib.layers.fully_connected(x, num_words,    
                                                                       None,    
                                                                       scope = decoding_scope,  
                                                                       weights_initializer=weights,    
                                                                       biases_initializer=biases
                                                                       )
        training_prediction = decode_training_set(encoder_state,    
                                                  decoder_cell, 
                                                  decoder_embedded_input,   
                                                  sequence_length,  
                                                  decoding_scope,   
                                                  output_function,
                                                  keep_prob,    
                                                  batch_size)
        decoding_scope.reuse_variables() 
        # can also be used for cross validation
        test_predictions = decode_test_set(encoder_state,   
                                           decoder_cell,    
                                           decoder_embedding_metrix, 
                                           word2int['<SOS>'],   
                                           word2int['<EOS>'], 
                                           sequence_length - 1, #not include the last token
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size
                                           ) 
    return training_prediction, test_predictions
        
#building the seq2seq model
#2 main part of seq2seq: encode RNN, decodde RNN
    

#encoder_embedding_size = the numnber of Dimemsion  of the embedded metrix for encoder
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, quesiotnswords2int):
    #the fuction will return training and test prediction
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,   
                                                              answers_num_words +1,  
                                                              encoder_embedding_size,
                                                              initializer= tf.random_uniform_initializer(0,1)  #argument(bound)
                                                              )
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    
    #get the preprocess target
    preprocessed_targets = preprocess_targets(targets, quesiotnswords2int, batch_size)
    
    decoder_embedding_metrix = tf.Variable(tf.random_uniform([questions_num_words+1,decoder_embedding_size],0,1))
    # use the decoder embedding metrix to get the decoder embedded input
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embedding_metrix, preprocessed_targets)
    training_prediction, test_prediction = decoder_rnn(decoder_embedded_input,  
                                                       decoder_embedding_metrix,    
                                                       encoder_state,   
                                                       questions_num_words, 
                                                       sequence_length,
                                                       rnn_size,
                                                       num_layers,
                                                       quesiotnswords2int,
                                                       keep_prob,
                                                       batch_size
                                                       )
    return training_prediction, test_prediction
    
    

    
 
###################### PART THREE Traininig the seq2seq medel ######################
    

#setting the hyper parameters
epochs = 100

batch_size = 64
rnn_size = 512
num_layers = 3
encoding_ebedding_size = 512
decoding_ebedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learninig_rate = 0.0001
keep_probability = 0.5

#Define a tensorflow session
#reset the tensorflow graph
tf.reset_default_graph()
session = tf.InteractiveSession()

#load the model input

inputs, targets, lr, keep_prob = model_inputs()

#setting the sequence_length<maximum length>
sequence_length = tf.placeholder_with_default(25, None, name= 'sequence_length')

#getting the shape of the inputs tensor
input_shape = tf.shape(inputs) 

#get the training and test prediction
training_prediction, test_prediction = seq2seq_model(tf.reverse(inputs,[-1]),   
                                                     targets,   
                                                     keep_prob, 
                                                     batch_size,    
                                                     sequence_length,
                                                     len(answerswords2int),
                                                     len(questionswords2int),
                                                     encoding_ebedding_size,
                                                     decoding_ebedding_size,
                                                     rnn_size,
                                                     num_layers,
                                                     questionswords2int,
                                                     )


#setting up the loss error, the optimizer and gradient clipping <preventing gradient exploding by clip the gradient in certain valeus>
#define a scope
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_prediction,  
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length])
                                                  )
    #get the Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #clip all our gradient
    gradients = optimizer.compute_gradients(loss_error) # the gradients is consisted of grad_tensor and grad_variable
    clipped_gradients = [(tf.clip_by_value(grad_tensor,-5.,5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
    
#padding the sequence wit the <PAD> token
#Quesition = ['who', 'are', 'you', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
#Answer = ['<SOS>', 'I', 'am', 'a'. 'robot', '.', '<EOS>', '<PAD>']
#Keep the Question and Answer have the same length

def apply_padding(batch_of_sequence, word2int):
    max_sequene_length = max([len(sequences) for sequences in batch_of_sequence])
    return [ sequence + [word2int['<PAD>']]* (max_sequene_length - len(sequence))for sequence in batch_of_sequence]

#Split the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions)// batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index: start_index+ batch_size]
        answers_in_batch = answers[start_index: start_index+ batch_size]
        # apply the padding
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, questionswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch
        
#splitting the question and answer into training and validation set
#splite into traning_question, training_answer, validation_question, validation_answer
        
training_validation_split = int (len(sorted_clean_questions)*0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]

## The main traininng part

#check the training loss, every 100 batches
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) -1 
total_training_loss_error= 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
#the for loop to do the training


writer = tf.summary.FileWriter("logs/", session.graph)

for epoch in range(1, epochs+1):
    #in each epoch we loop all the batches
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error],   
                                                   {inputs : padded_questions_in_batch,
                                                    targets: padded_answers_in_batch,   
                                                    lr : learning_rate,
                                                    sequence_length : padded_answers_in_batch.shape[1], 
                                                    keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training loss error: {:>6.3f}, Training tiem on 100 batches: {:d} seconds'.format(epoch,   
                  epochs,   
                  batch_index,  
                  len(training_questions)//batch_size,  
                  total_training_loss_error / batch_index_check_training_loss,  
                  int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0        
            
        #validation set
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error,   
                                                          {inputs : padded_questions_in_batch,
                                                          targets: padded_answers_in_batch,   
                                                          lr : learning_rate,
                                                          sequence_length : padded_answers_in_batch.shape[1], 
                                                          keep_prob: 1})
                total_validation_loss_error += batch_training_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss = total_validation_loss_error / (len(validation_questions)/ batch_size)
            print("Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds".format(average_validation_loss, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learninig_rate:
                learning_rate = min_learninig_rate
            #early stopping
            list_validation_loss_error.append(average_validation_loss)
            if average_validation_loss <= min (list_validation_loss_error):
                print("I speak better now")
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Do not speak better")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("Can't be improved")
        break
print("Game over!")
            
            
#Loading the weight and Running the session
checkpoint ="./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer)
saver = tf.train.Saver()
saver.restore(session, checkpoint)

#Converting the question from strings to list of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word['<OUT>']) for word in question.split()]





while (True):
    question = input ("You: ")
    if question == "Goodbye":
        break
    question = conversation(question, questionswords2int)
    question = question + [questionswords2int['<PAD>']]* (20-len(question))
    fake_batch = np.zeros((batch_size, 20))
    fake_batch[0] = question
    predicted_answer = session.run(test_prediction, {inputs: fake_batch,    
                                                     keep_prob : 0.5    
                                                     })[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersint2word[i] == 'i':
            token = 'I'
        elif answersint2word[i] == '<EOS>':
            token = '.'
        elif answersint2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' '+ answersint2word[i]
        answer += token 
        if token == '.':
            break
    print('ChatBot: ' + answer)
            
                                                      

        
    














