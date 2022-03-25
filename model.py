#-----------------------------------
# 论文提供的原版网络结构
#-----------------------------------
from tensorflow.keras.layers import Conv2D,Flatten,MaxPooling2D,Input,BatchNormalization,Dropout,Dense
from tensorflow.keras.models import Model
def create_model(input_shape = (48,48,1),num_classes=7):

    input = Input(shape=input_shape)
    x = Conv2D(filters=256,kernel_size=3,activation='relu',padding='same')(input)

    x = Conv2D(filters=512,kernel_size=3,activation='relu',padding='same')(x)
    x = BatchNormalization()(x)

    #
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filters=384,kernel_size=3,activation='relu',padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filters=192,kernel_size=3,activation='relu',padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.4)(x)


    x = Conv2D(filters=384,kernel_size=3,activation='relu',padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)

    x = Dense(256,activation='relu')(x)
    x = BatchNormalization()(x)

    x = Dropout(0.3)(x)
    x = Dense(num_classes,activation='softmax')(x)

    return Model(input,x,name='fer_model')


if __name__=='__main__':
    model = create_model()
    model.summary()





