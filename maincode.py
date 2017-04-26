# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#voor tensorboard: tensorboard --logdir=/tmp/tensorflow/mnist/logs/fully_connected_feed
#Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time
#import numpy as np
import scipy
from scipy import ndimage # voor het inladen van plaatje
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import functies 
import importeren
import random
import importerentest




# Basic model parameters as external flags.
FLAGS = None
"""in FLAGS worden gewoon bepaalde parameters opgeslagen, zoals bijvoorbeeld FLAGS.max_steps"""
plaatjes_breedte = functies.IMAGE_BREEDTE
plaatjes_hoogte = functies.IMAGE_HOOGTE
batch_size = 20
num_examples = 60 #ik wee tniet precies wat dit is, heb het nu maar gewoon even een waarde gegeven
fulconunits = 200 #ipv 1024
verspringing = 2
vakbreedte = 5

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         functies.IMAGE_PIXELS))
  
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size)) # dit geeft [2,], terwijl de echte labels tensor [2,1] heeft
  #bij het originele bestand hebben de échte labels ook een shape van [batchsize,] Fake data? Sad!
  print('images_placeholder')
  print(images_placeholder.shape)
  print('labels_placeholder')
  print(labels_placeholder.shape)
  return images_placeholder, labels_placeholder


def fill_feed_dict( images_pl, labels_pl):
  """Fills the feed_dict for training the given step. Het is een zogenaamde 'dictionary object'
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = importeren.nextbatch(batch_size,plaatjes_breedte, plaatjes_hoogte)
  #images_feed, labels_feed = data_s(FLAGS.batch_size, FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

def fill_feed_dicttest( images_pl, labels_pl):
  """Fills the feed_dict for training the given step. Het is een zogenaamde 'dictionary object'
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = importerentest.nextbatch(batch_size,plaatjes_breedte, plaatjes_hoogte)
  #images_feed, labels_feed = data_s(FLAGS.batch_size, FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_evaltest(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = num_examples // batch_size
  
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dicttest(   #hier stond eerst data_set
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def do_evaltrain(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = num_examples // batch_size
  
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(   #hier stond eerst data_set
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def run_training():
  """Train MNIST for a number of steps."""
   

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default(): #in deze graph worden de operations opgeslagen/verzameld
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(
        batch_size)


   
    # Build a Graph that computes predictions from the inference model.
    logits = functies.convinference(images_placeholder, fulconunits,vakbreedte,verspringing)  #convolution inferencefunctie
    #logits = functies.inference(images_placeholder, FLAGS.hidden1,FLAGS.hidden2) #oude inference functie

    # Add to the Graph the Ops for loss calculation.
    loss = functies.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = functies.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = functies.evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    
    


    """Tot hier was dus allemaal voorbereiding van alles in de Graph. Nu gaat de graph pas echt gerund worden"""





    # Create a session for running Ops on the Graph.
    # Once all of the build preparation has been completed and all of the necessary ops generated, a tf.Session is created for running the graph.
    sess = tf.Session() 
 
    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables. Hier worden alleen de variables geinitialiseerd. 
    #Het runnen van de training gebeurt hieronder pas bij die for loop.
    sess.run(init)

    # Start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      
      feed_dict = fill_feed_dict( 
                                 images_placeholder,
                                 labels_placeholder)
      
      #print('hieronder printen we feed_dict op dit moment met placeholders')
      #print(feed_dict)


      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)  

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 5 == 0: #dit print elke 100 steps een update van de loss
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict) #"summary" was een merge van alle summaries, hierboven al in de Graph geinitialiseerd
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()


      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 5 == 0 or (step + 1) == FLAGS.max_steps:
        
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_evaltrain(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder) #hier moet iets veranderen, want data_sets bestaan niet meer
        # Evaluate against the validation set.
        """
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                validationset) #hier moet iets veranderen, want data_sets bestaan niet meer
       """
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_evaltest(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder)#hier moet iets veranderen, want data_sets bestaan niet meer
 

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01, #was 0.01
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=100,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/tmp/tensorflow/mnist/logs/learning0.01',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  
  
  
