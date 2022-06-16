"""
Original work by Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com, https://github.com/vickysun5/SmartProcessAnalytics
Modified by Pedro Seber, https://github.com/PedroSeber/SmartProcessAnalytics
"""
import numpy as np
import tensorflow as tf

# Generate batch data
def gen_batch(raw_x, raw_y, raw_yp, batch_size, num_steps, epoch_overlap):
    data_length = len(raw_x)
    dx = np.shape(raw_x)[1]
    dy = np.shape(raw_y)[1]
    dyp = np.shape(raw_yp)[1]
    
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length,dx], dtype= np.float32)
    data_y = np.zeros([batch_size, batch_partition_length,dy], dtype= np.float32)
    data_yp = np.zeros([batch_size, batch_partition_length,dyp], dtype= np.float32)
    
    for idx in range(batch_size):
        data_x[idx] = raw_x[batch_partition_length*idx : batch_partition_length*(idx+1)]
        data_y[idx] = raw_y[batch_partition_length*idx : batch_partition_length*(idx+1)] 
        data_yp[idx] = raw_yp[batch_partition_length*idx : batch_partition_length*(idx+1)] 
    
    if epoch_overlap is None:
        epoch_size = batch_partition_length // num_steps
        for idx in range(epoch_size):
            x = data_x[:, idx*num_steps : (idx+1)*num_steps]
            y = data_y[:, idx*num_steps : (idx+1)*num_steps]
            yp = data_yp[:, idx*num_steps : (idx+1)*num_steps]
            yield (x, y, yp)
    else:
        epoch_size = (batch_partition_length - num_steps + 1)//(epoch_overlap+1)
        for idx in range(epoch_size):
            x = data_x[:, idx*(epoch_overlap+1) : idx*(epoch_overlap+1) + num_steps]
            y = data_y[:, idx*(epoch_overlap+1) : idx*(epoch_overlap+1) + num_steps]
            yp = data_yp[:, idx*(epoch_overlap+1) : idx*(epoch_overlap+1) + num_steps]
            yield (x, y, yp)

# Generate batch data for multiple series
def gen_batch_multi(raw_x, raw_y, timeindex, batch_size, num_steps, epoch_overlap):
    cum = 0
    num_series = len(timeindex)
    for s in range(num_series):
        num = np.shape(timeindex[s+1])[0]       
        x = raw_x[cum : cum+num]
        y = raw_y[cum : cum+num]
        yp = np.insert(y, 0, 0, axis = 0)[:-1]
        data_length = len(x)
        dx = np.shape(x)[1]
        dy = np.shape(y)[1]
        dyp = np.shape(yp)[1]
        
        batch_partition_length = data_length // batch_size
        data_x = np.zeros([batch_size, batch_partition_length,dx], dtype= np.float32)
        data_y = np.zeros([batch_size, batch_partition_length,dy], dtype= np.float32)
        data_yp = np.zeros([batch_size, batch_partition_length,dyp], dtype= np.float32)
        
        for idx in range(batch_size):
            data_x[idx] = x[batch_partition_length*idx : batch_partition_length*(idx+1)]
            data_y[idx] = y[batch_partition_length*idx : batch_partition_length*(idx+1)] 
            data_yp[idx] = yp[batch_partition_length*idx : batch_partition_length*(idx+1)] 
        
        if epoch_overlap is None:
            epoch_size = batch_partition_length // num_steps
            for idx in range(epoch_size):
                x = data_x[:, idx*num_steps : (idx+1)*num_steps]
                y = data_y[:, idx*num_steps : (idx+1)*num_steps]
                yp = data_yp[:, idx*num_steps : (idx+1)*num_steps]
                yield (x, y, yp, s)
        else:
            epoch_size = (batch_partition_length - num_steps + 1)//(epoch_overlap+1)
            for idx in range(epoch_size):
                x = data_x[:, idx*(epoch_overlap+1) : idx*(epoch_overlap+1)+num_steps]
                y = data_y[:, idx*(epoch_overlap+1) : idx*(epoch_overlap+1)+num_steps]
                yp = data_yp[:, idx*(epoch_overlap+1) : idx*(epoch_overlap+1)+num_steps]
                yield (x, y, yp, s)
        
        cum += num 

# Generate batch data for kstep prediction
def gen_batch_kstep(raw_x, raw_y,raw_yp, rnn_state, batch_size, num_steps, epoch_overlap):
    data_length = len(raw_x)
    dx = np.shape(raw_x)[1]
    dy = np.shape(raw_y)[1]
    dyp = np.shape(raw_yp)[1]
    ds = np.shape(rnn_state)[1]
    
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length,dx], dtype= np.float32)
    data_y = np.zeros([batch_size, batch_partition_length,dy], dtype= np.float32)
    data_yp = np.zeros([batch_size, batch_partition_length,dyp], dtype= np.float32)
    data_s = np.zeros([batch_size, batch_partition_length,ds], dtype= np.float32)
    
    for idx in range(batch_size):
        data_x[idx] = raw_x[batch_partition_length*idx : batch_partition_length*(idx+1)]
        data_y[idx] = raw_y[batch_partition_length*idx : batch_partition_length*(idx+1)]
        data_yp[idx] = raw_yp[batch_partition_length*idx : batch_partition_length*(idx+1)] 
        data_s[idx] = rnn_state[batch_partition_length*idx : batch_partition_length*(idx+1)] 
  
    if epoch_overlap is None:
        epoch_size = batch_partition_length // num_steps
        for idx in range(epoch_size):
            x = data_x[:, idx*num_steps : (idx+1)*num_steps]
            y = data_y[:, idx*num_steps : (idx+1)*num_steps]
            yp = data_yp[:, idx*num_steps : (idx+1)*num_steps]
            s = data_s[:, idx*num_steps : (idx+1)*num_steps]
            yield (x, y, yp, s)
    else:
        epoch_size = (batch_partition_length - num_steps + 1)//(epoch_overlap+1)
        for idx in range(epoch_size):
            x = data_x[:, idx*(epoch_overlap+1) : idx*(epoch_overlap+1) + num_steps]
            y = data_y[:, idx*(epoch_overlap+1) : idx*(epoch_overlap+1) + num_steps]
            yp = data_yp[:, idx*(epoch_overlap+1) : idx*(epoch_overlap+1) + num_steps]
            s = data_s[:, idx*(epoch_overlap+1) : idx*(epoch_overlap+1) + num_steps]
            yield (x, y, yp, s)
            
# Generate batch data for kstep prediction
def gen_batch_kstep_layer(raw_x, raw_y, raw_yp, rnn_state):
    data_length = len(raw_x)
    dx = np.shape(raw_x)[1]
    dy = np.shape(raw_y)[1]
    dyp = np.shape(raw_yp)[1]
    
    num_layers = len(rnn_state)
    batch_size = data_length
    batch_partition_length = 1

    data_x = np.zeros([batch_size, batch_partition_length, dx], dtype= np.float32)
    data_y = np.zeros([batch_size, batch_partition_length, dy], dtype= np.float32)
    data_yp = np.zeros([batch_size, batch_partition_length, dyp], dtype= np.float32)
    final_data_s = ()

    for idx in range(batch_size):
        data_x[idx] = raw_x[batch_partition_length*idx : batch_partition_length*(idx+1)]
        data_y[idx] = raw_y[batch_partition_length*idx : batch_partition_length*(idx+1)]
        data_yp[idx] = raw_yp[batch_partition_length*idx : batch_partition_length*(idx+1)]
    for idx in range(num_layers): 
          final_data_s += (rnn_state[idx][:-1],)
    yield (data_x, data_y, data_yp, final_data_s)
            
def gen_epochs(raw_data_x, raw_data_y, raw_data_yp, num_epochs, num_steps, batch_size, epoch_overlap):
    for i in range(int(num_epochs)):
        yield gen_batch(raw_data_x, raw_data_y, raw_data_yp, batch_size, num_steps, epoch_overlap)

def gen_epochs_multi(raw_data_x, raw_data_y, timeindex, num_epochs, num_steps, batch_size, epoch_overlap):
    for i in range(int(num_epochs)):
        yield gen_batch_multi(raw_data_x, raw_data_y, timeindex, batch_size, num_steps, epoch_overlap)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

# Define RNN graph
def build_multilayer_rnn_graph_with_dynamic_rnn(cell_type, RNN_layers, activation, num_steps, x_num_features, y_num_features , learning_rate, lambda_l2_reg, random_seed = 0):
    reset_graph()
    tf.set_random_seed(random_seed) # For reproducibility
    x_num_features += y_num_features
    
    # Graph inputs
    batch_size = tf.keras.Input([], dtype = tf.int32, name = 'batch_size') # TODO: test whether change to keras.Input was successful
    x = tf.keras.Input([None, num_steps, x_num_features], dtype = tf.float32, name='x')
    y = tf.keras.Input([None, num_steps, y_num_features], dtype = tf.float32, name='y')
    input_prob = tf.keras.Input(1, name='input_prob', dtype = tf.float32)
    state_prob = tf.keras.Input(1, name='state_prob', dtype = tf.float32)
    output_prob = tf.keras.Input(1, name='output_prob', dtype = tf.float32)
    rnn_inputs = x

    # Define a single cell with variational dropout
    def get_a_cell(first_layer, input_prob, state_prob, num_input):
        # First, define the cell type
        if cell_type.casefold() == 'lstm':
            cell_function = tf.nn.rnn_cell.LSTMCell
        elif cell_type.casefold() == 'gru':
            cell_function = tf.nn.rnn_cell.GRUCell
        else: # Basic RNN
            cell_function = tf.nn.rnn_cell.BasicRNNCell
        # Then, define the activation function and create a cell
        if activation.casefold() == 'linear':
            my_cell = cell_function(first_layer, activation = tf.identity)
        elif activation.casefold() == 'relu':
            my_cell = cell_function(first_layer, activation = tf.nn.relu)
        elif activation.casefold() == 'tanh':
            my_cell = cell_function(first_layer)
        elif activation.casefold() == 'sigmoid':
            my_cell = cell_function(first_layer, activation = tf.math.sigmoid)
        else:
            raise ValueError(f'Activation must be in {{"linear", "relu", "tanh", "sigmoid"}}, but you passed {activation}')
        # Finally, create a dropout wrapper
        cell_drop = tf.contrib.rnn.DropoutWrapper(my_cell, variational_recurrent = True, dtype = tf.float32, input_size = num_input, input_keep_prob = input_prob, state_keep_prob = state_prob)
        return cell_drop

    # Create the full RNN and put it into a dropout wrapper
    cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(layer, input_prob, state_prob, x_num_features if layer == RNN_layers[0] else layer) for layer in RNN_layers])
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, variational_recurrent = True, dtype = tf.float32, input_size = x_num_features, output_keep_prob = output_prob)
    init_state = cell.zero_state(batch_size, dtype = tf.float32)
    # Build dynamic graph
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell = cell, inputs = rnn_inputs, initial_state = init_state)

    # Final softmax for prediction
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [RNN_layers[-1], y_num_features])
        b = tf.get_variable('b', [y_num_features], initializer = tf.constant_initializer(0.0))
    rnn_outputs = tf.reshape(rnn_outputs, [-1, RNN_layers[-1]])
    predictions = tf.matmul(rnn_outputs, W) + b
    yy = tf.reshape(y, [-1, y_num_features])   # TODO: Should we have this double reshape here and in the MSE?
    loss = tf.keras.metrics.mean_square_error(tf.reshape(predictions,[-1]), tf.reshape(yy,[-1]))

    # Adding regularization
    if lambda_l2_reg > 0 :
        cell_l2 = tf.reduce_sum([tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if not ("noreg" in tf_var.name or "Bias" in tf_var.name)])
        Predict_l2 = tf.nn.l2_loss(W) #+ tf.nn.l2_loss(b)
        total_loss = tf.reduce_sum( loss + lambda_l2_reg*tf.reduce_sum(cell_l2 + Predict_l2) )
    else:
        total_loss = loss

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    return dict(x = x, y = y, batch_size = batch_size, input_prob = input_prob, state_prob = state_prob,
                output_prob = output_prob, init_state = init_state, final_state = final_state, rnn_outputs = rnn_outputs,
                total_loss = total_loss, loss = loss, train_step = train_step, preds = predictions, saver= tf.train.Saver())
    
# Train RNN graph
def train_rnn(raw_data_x, raw_data_y, val_data_x, val_data_y, g, num_epochs, num_steps, batch_size, input_prob, output_prob, state_prob, epoch_before_val = 50,
                max_checks_without_progress = 50, epoch_overlap = None, verbose = True, save = False):
    with tf.Session() as sess:
        # initialize the variables
        sess.run(tf.global_variables_initializer())
        raw_data_yp = np.insert(raw_data_y, 0, 0, axis = 0)[:-1]
        val_data_yp = np.insert(val_data_y, 0, 0, axis = 0)[:-1]

        variable_shapes = [v.get_shape() for v in tf.trainable_variables()]
        parameter_num = 0
        for shape in variable_shapes: # TODO: can probably change this to a cumsum / cumprod
            parameter_num += shape[0]*shape[1] if np.size(shape)>1 else shape[0]

        training_losses = []
        val_losses = []
        checks_without_progress = 0 # For early stopping
        best_loss = np.infty
        
        for idx, epoch in enumerate(gen_epochs(raw_data_x,raw_data_y,raw_data_yp,num_epochs, num_steps, batch_size, epoch_overlap)):
            training_loss = 0
            training_state = None
            for steps, (X, Y, YP) in enumerate(epoch):
                feed_dict = {g['x']: np.dstack((X,YP)), g['y']: Y, g['batch_size']:batch_size, g['input_prob']: input_prob ,g['output_prob']: output_prob,g['state_prob']:state_prob}
                # Continue to feed in if in the same class
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['loss'], g['final_state'], g['train_step']], feed_dict = feed_dict)
                training_loss += training_loss_

            if np.isnan(training_loss_):
                raise ValueError(f'The training loss after epoch {steps} was NaN')
            if verbose and not(idx%100):
                print(f"Average training loss for Epoch {idx}: {training_loss/(steps+1)}")
            training_losses.append(training_loss / (steps+1))
            
            # Validation
            if idx > epoch_before_val:
                val_loss = 0
                val_state = None
                for steps_val, (X_val, Y_val, YP_val) in enumerate(gen_batch(val_data_x, val_data_y, val_data_yp, batch_size, num_steps, epoch_overlap)):
                    feed_dict_val = {g['x']: np.dstack((X_val,YP_val)), g['y']: Y_val, g['batch_size']: batch_size, g['input_prob']: 1 , g['output_prob']: 1, g['state_prob']: 1}
                    # Continue to feed in if in the same class
                    if val_state is not None:
                        feed_dict_val[g['init_state']] = val_state
                    val_loss_, val_state = sess.run([g['loss'],  g['final_state']],feed_dict=feed_dict_val)
                    val_loss += val_loss_
                    
                val_loss = val_loss/(steps_val+1)
                val_losses.append(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    checks_without_progress = 0
                    g['saver'].save(sess, save)
                else:
                    checks_without_progress += 1
                    if checks_without_progress > max_checks_without_progress:
                        print("Early stopping!")
                        break
        else:
            print('Max number of train epochs reached')
            
        if isinstance(save, str):
            g['saver'].save(sess, save)
        training_losses = np.array(training_losses)
        val_losses = np.array(val_losses)
    return (training_losses, val_losses, int(parameter_num))


# Train RNN graph for multiple series
def train_rnn_multi(raw_data_x, raw_data_y, val_data_x, val_data_y, timeindex_train, timeindex_val, g, num_epochs, num_steps, batch_size, input_prob, output_prob,
                    state_prob, epoch_before_val = 50, max_checks_without_progress = 50, epoch_overlap = None, verbose = True, save = False):
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
        variable_shapes = [v.get_shape() for v in tf.trainable_variables()]
        parameter_num = 0
        for shape in variable_shapes: # TODO: can probably change this to a cumsum / cumprod
            parameter_num += shape[0]*shape[1] if np.size(shape)>1 else shape[0]

        training_losses = []
        val_losses = []
        checks_without_progress = 0 # For early stopping
        best_loss = np.infty
        
        for idx, epoch in enumerate(gen_epochs_multi(raw_data_x,raw_data_y, timeindex_train, num_epochs, num_steps, batch_size, epoch_overlap)):
            training_loss = 0
            s_threshold = 0
            training_state = None
            for steps, (X, Y, YP, s) in enumerate(epoch):
                feed_dict = {g['x']: np.dstack((X,YP)), g['y']: Y, g['batch_size']:batch_size, g['input_prob']: input_prob ,g['output_prob']: output_prob,g['state_prob']:state_prob}
                # Start to feed 0 initial for a new set of class
                if s == s_threshold:
                    s_threshold += 1
                    training_state = None
                # Continue to feed in if in the same class
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['loss'], g['final_state'], g['train_step']], feed_dict = feed_dict)
                training_loss += training_loss_

            if verbose and not(idx%100):
                print(f"Average training loss for Epoch {idx}: {training_loss/(steps+1)}")
            training_losses.append(training_loss / (steps+1))
            
            # Test on validation set
            if idx > epoch_before_val:
                val_loss = 0
                s_val_threshold = 0
                val_state = None
                for steps_val, (X_val, Y_val, YP_val, s_val) in enumerate(gen_batch_multi(val_data_x, val_data_y, timeindex_val, batch_size, num_steps, epoch_overlap)):
                    feed_dict_val = {g['x']: np.dstack((X_val,YP_val)), g['y']: Y_val, g['batch_size']:batch_size, g['input_prob']: 1 , g['output_prob']: 1, g['state_prob']: 1}
                    # Start to feed 0 initial for a new set of class
                    if s_val == s_val_threshold:
                        s_val_threshold += 1
                        val_state = None
                    # Continue to feed in if in the same class
                    if val_state is not None:
                        feed_dict_val[g['init_state']] = val_state
                    val_loss_,val_state = sess.run([g['loss'],  g['final_state']],feed_dict=feed_dict_val)
                    val_loss += val_loss_
                    
                val_loss = val_loss/(steps_val+1)
                val_losses.append(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    checks_without_progress = 0
                    g['saver'].save(sess, save)
                else:
                    checks_without_progress += 1
                    if checks_without_progress > max_checks_without_progress:
                        print("Early stopping!")
                        break
        else:
            print("Max number of train epochs reached")
            
        if isinstance(save, str):
            g['saver'].save(sess, save)
        training_losses = np.array(training_losses)
        val_losses = np.array(val_losses)
    return (training_losses,val_losses, int(parameter_num))

# Test RNN graph 0 step
def test_rnn(test_data_x, test_data_y, g, checkpoint, input_prob, output_prob, state_prob, num_test):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_data_yp = np.insert(test_data_y,0,0,axis=0)[:-1]
        # Read the trained graph
        g['saver'].restore(sess, checkpoint)
        # Run the test points
        for index, (X, Y, YP) in enumerate(gen_batch(test_data_x, test_data_y, test_data_yp, 1, num_test, None)):
            feed_dict={g['x']: np.dstack((X,YP)), g['y']: Y, g['batch_size']: 1, g['input_prob']: input_prob, g['output_prob']: output_prob, g['state_prob']: state_prob}
            preds, rnn_outputs = sess.run([g['preds'], g['rnn_outputs']], feed_dict)
        
        loss = np.sum((preds[1:]-test_data_y[1:])**2,axis=0)/(test_data_y.shape[0]-1)
    return (preds,loss,rnn_outputs)

# Test RNN graph 0 step for multiplayer afterwards
def test_rnn_layer(test_data_x, test_data_y, g, checkpoint, input_prob, output_prob, state_prob, num_test, num_layers):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_data_yp = np.insert(test_data_y,0,0,axis=0)[:-1]
        final = {}
        # Read the trained graph
        g['saver'].restore(sess, checkpoint)
        # Run the test points
        for index, (X, Y, YP) in enumerate(gen_batch(test_data_x, test_data_y, test_data_yp, 1, 1, None)):
            feed_dict = {g['x']: np.dstack((X,YP)), g['y']: Y, g['batch_size']: 1, g['input_prob']: input_prob, g['output_prob']: output_prob, g['state_prob']: state_prob}
            if index > 0:
                feed_dict[g['init_state']] = rnn_outputs

            preds, rnn_outputs = sess.run([g['preds'], g['final_state']], feed_dict)
            if index > 0:
                final_preds = np.vstack((final_preds, preds))
            else:
                final_preds = preds
                
            for layer_idx in range(num_layers):
                if index > 0:
                    final[layer_idx] = np.vstack( (final[layer_idx], rnn_outputs[layer_idx]) )
                else:
                    final[layer_idx] = rnn_outputs[layer_idx]
                    
        final_inter_state = ()
        for idx in range(num_layers):
            final_inter_state += (final[idx],) 
            
        loss = np.sum((final_preds[1:]-test_data_y[1:])**2,axis=0)/(test_data_y.shape[0]-1)
    return (final_preds, loss, final_inter_state)

# Test RNN graph single layer
def test_rnn_kstep(test_data_x, test_data_y, preds, rnn_outputs, g, checkpoint, input_prob, output_prob, state_prob, num_test, kstep = 3):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = {}
        # Read the trained graph
        g['saver'].restore(sess, checkpoint)
        losses = []
        for step_num in range(1, kstep+1):
            for index, (X, Y, YP, S) in enumerate(gen_batch_kstep(test_data_x[step_num:], test_data_y[step_num:], preds[:-1], rnn_outputs[:-1], num_test - step_num, 1, None)):
                feed_dict = {g['x']: np.dstack((X,YP)), g['y']: Y, g['init_state']: np.squeeze(S), g['batch_size']: num_test, g['input_prob']: input_prob, g['output_prob']: output_prob, g['state_prob']: state_prob}
                preds, rnn_outputs = sess.run([g['preds'], g['rnn_outputs']], feed_dict)
                loss = np.sum((preds[1:] - test_data_y[1+step_num:])**2, axis = 0)/test_data_y[1+step_num:].shape[0]
                result[step_num] = preds
                losses.append(loss)
    return (result,losses)

# Test RNN graph multi layer
def test_rnn_kstep_layer(test_data_x, test_data_y, preds, rnn_outputs, g, checkpoint, input_prob, output_prob, state_prob, num_test, kstep = 3):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result= {}
        # Read the trained graph
        g['saver'].restore(sess, checkpoint)
        losses = []
        for step_num in range(1, kstep+1):
            for index, (X, Y, YP, S) in enumerate(gen_batch_kstep_layer(test_data_x[step_num:], test_data_y[step_num:], preds[:-1], rnn_outputs)):
                feed_dict = {g['x']: np.dstack((X,YP)), g['y']: Y, g['init_state']: S, g['batch_size']: num_test, g['input_prob']: input_prob, g['output_prob']: output_prob, g['state_prob']: state_prob}
                preds, rnn_outputs = sess.run([g['preds'], g['final_state']], feed_dict)
                loss = np.sum((preds[1:] - test_data_y[step_num+1:])**2, axis = 0)/test_data_y[step_num+1:].shape[0]
                result[step_num] = preds
                losses.append(loss)
    return (result,losses)

