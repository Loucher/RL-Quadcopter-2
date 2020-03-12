import tensorflow as tf


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = tf.keras.Input(shape=(self.state_size,), name='states')
        actions = tf.keras.layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = tf.keras.layers.Dense(
            units=300,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activity_regularizer=tf.keras.regularizers.l1(0.01)
        )(states)
        net_states = tf.keras.layers.Dense(
            units=400,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activity_regularizer=tf.keras.regularizers.l1(0.01)
        )(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = tf.keras.layers.Dense(
            units=300,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activity_regularizer=tf.keras.regularizers.l1(0.01)
        )(actions)
        net_actions = tf.keras.layers.Dense(
            units=400,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activity_regularizer=tf.keras.regularizers.l1(0.01)
        )(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = tf.keras.layers.Add()([net_states, net_actions])
        net = tf.keras.layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to produce action values (Q values)
        Q_values = tf.keras.layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = tf.keras.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = tf.keras.backend.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = tf.keras.backend.function(
            inputs=[*self.model.input, tf.keras.backend.learning_phase()],
            outputs=action_gradients)
