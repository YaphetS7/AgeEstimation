import telebot
from telebot.types import InputMediaPhoto

bot = telebot.TeleBot('') # insert ur token here


def preprocess_photo(photo, uid, postfix):
    fileID = photo.photo[-1].file_id

    file_info = bot.get_file(fileID)

    downloaded_file = bot.download_file(file_info.file_path)
    path = '../' + str(uid) + postfix + ".jpg"
    with open(path, 'wb') as new_file:
        new_file.write(downloaded_file)
    return path#open(str(uid) + postfix + ".jpg", 'rb')


@bot.message_handler(commands=['start'])
def start_message(message):
    user_id = message.from_user.id
    bot.send_message(message.chat.id, 'Привет! Тут можно оценить свой возраст\nНапиши /help, если ты не знаешь как со мной работать.')


@bot.message_handler(commands=['help'])
def start_message(message):
    user_id = message.from_user.id
    bot.send_message(message.chat.id, 'Отправь мне фотографию, я оценю твой возраст! :)')



@bot.message_handler(content_types=['photo'])
def send_photo(photo):
    user_id = photo.from_user.id

    img_path = preprocess_photo(photo, user_id, 'age')

    p = inference(img_path, user_id)
    if type(p[0]) == str:
        message = p
    else:
        p1, p2 = p
        message = f'alignment face prediction: {p1}\nnot alignment face prediction: {p2}'
    

    bot.send_message(photo.chat.id, message)



bot.polling()