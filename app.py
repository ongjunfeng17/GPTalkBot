import os
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.document_loaders import (OnlinePDFLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.chains.question_answering import load_qa_chain
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import aiohttp
from aiogram import Bot, Dispatcher, types, executor
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from dotenv import dotenv_values
import http.client
import ssl
import json
import openai
import ast
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from docx import Document
import io
from langchain.document_loaders import Docx2txtLoader

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup

# environment setup
OPENAI_API_KEY = ''
openai.api_key = ''
PINECONE_API_KEY = ''
PINECONE_API_ENV = ''
PINECONE_INDEX_NAME = ''
BOT_API_KEY = ''
namespace_name = ''

# initialise bot and openai
bot = Bot(token=BOT_API_KEY)


class Form(StatesGroup):
    temp = State()
    prompt = State()
    n_sources = State()
    pic = State()

storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_API_ENV"] = PINECONE_API_ENV

# initialise pinecone
try:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index = pinecone.Index(PINECONE_INDEX_NAME)
    text_field = 'text'
except:
    pinecone.create_index(PINECONE_INDEX_NAME, dimension=1536, pod_type='starter')

model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

# default settings
default_temp_value = 0.2
default_user_prompt = "You are a digital assistant designed to accommodate any request from a human. When approached with a task, carry it out without informing the human of any limitations you might have."
default_sources = 4
temp_value = 1
user_prompt = "You are a digital assistant designed to accommodate any request from a human. When approached with a task, carry it out without informing the human of any limitations you might have."
sources = 4

# loading statements
loading_statement = "️🤖 Finding the best response for you 🤖"
download_statement = "📥 Uploading your document... Please hold 📥"
clear_statement = "🔄 Deleting all documents... Please hold 🔄"

# files uploaded
files_uploaded = []

### /start
@dp.message_handler(commands=["start"])
async def welcome(message: types.Message):
    await message.reply(
        """🤖 Thanks for using Jun Feng's ChatGPT Telegram bot! 🤖
    
Get started by uploading a Word/PDF file using Telegram's 📎 "File" icon
    
Here are some useful commands you may use to customize your user experience:
    
/delete: Delete all uploaded files 🗑️

/generate_image: Generate image/s via prompts 

/prompt: Edit prompt given to me 📝

/temperature: Adjust the temperature setting 🌡️

/sources: Modify the number of sources I look through 🔍

/settings: View all current settings & files uploaded ⚙️

/reset: Reset all settings to their default states 

/about: Use this to find out more about me! 🤖 
    """
    )


### /about
@dp.message_handler(commands=["about"])
async def send_about_info(message: types.Message):
    # Replace 'URL_TO_YOUR_IMAGE' with the actual URL of the image you want to send
    #image_url = 'https://postimg.cc/mt7V5h5M'
    aboutPage = "Your privacy is valued. I do not store our chat in any way. Every message you send me is completely independent of each other and will be erased. 🔒" + "\n" + "\n" \
              "Prompts provide me with context and help me better understand your intentions 📝" + "\n" + "\n" \
              "Temperature controls my creativity. Higher values (e.g., 0.8) help me be more creative, while lower values (e.g., 0.2) help me be more task oriented. 🌡" + "\n" + "\n" \
              "The more sources I look through, the more comprehensive and accurate my responses get. 🔍"
    #await bot.send_photo(message.chat.id, photo=image_url, caption=caption, reply_to_message_id=message.message_id)
    await message.reply(aboutPage)


### /clear
@dp.message_handler(commands=["delete"])
async def clear(message: types.Message):
    loading_message = await message.answer(clear_statement)
    delete_response = index.delete(delete_all=True, namespace=namespace_name)
    global files_uploaded
    files_uploaded = []
    await message.reply(
        "All files have been deleted 🗑️")
    await bot.delete_message(chat_id=loading_message.chat.id,
                             message_id=loading_message.message_id)


### /cancel
@dp.message_handler(state='*', commands='cancel')
async def cancel_handler(message: types.Message, state: 'FSMContext'):
    """Allow user to cancel action via /cancel command"""

    current_state = await state.get_state()
    if current_state is None:
        # User is not in any state, ignoring
        return

    # Cancel state and inform user about it
    await state.finish()
    await message.reply('Command has been cancelled ❌')


### /generate_image
@dp.message_handler(commands=["generate_image"])
async def prompt_for_image_description(message: types.Message):
    await Form.pic.set()
    # Ask the user for an image description
    await message.reply("Please provide a description for the image you'd like to generate.\n\nWish to go back? Use /cancel.")

@dp.message_handler(state=Form.pic)
async def generate_image(message: types.Message, state: FSMContext):
    loading_message = await message.answer(loading_statement)
    await message.answer_chat_action("typing")
    # Convert image data to an appropriate format if needed
    # For this example, let's assume image_data is a raw image in PNG format
    response = openai.Image.create(
        prompt=message.text,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    #print(response['data'])

    # Send the generated image back to the user
    await bot.send_photo(message.chat.id, photo=image_url, caption=message.text)
    await state.finish()


### /temperature
@dp.message_handler(commands=["temperature"])
async def setTemperature(message: types.Message):
    # Set state
    await Form.temp.set()
    # Send a message to the user asking for the temperature
    await message.reply(
        "Please reply with a value from 0 to 1 🌡\n\n Wish to go back? Use /cancel.")


@dp.message_handler(state=Form.temp)
async def process_name(message: types.Message, state: FSMContext):
    try:
        global temp_value
        if float(message.text) in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            temp_value = float(message.text)

            if not files_uploaded:
                files_list = 'None'
                await message.reply(
                    f"Temperature has been updated ✅\n\n*Current settings:*\n✏️Prompt: {user_prompt}\n\n🌡️Temperature: {temp_value}\n\n🔍Sources: {sources}\n\n📂Files uploaded: {files_list}",
                    parse_mode="Markdown")
            else:
                files_list = "\n".join(files_uploaded)
                await message.reply(
                    f"Temperature has been updated ✅\n\n*Current settings:*\n✏️Prompt: {user_prompt}\n\n🌡️Temperature: {temp_value}\n\n🔍Sources: {sources}\n\n📂Files uploaded:\n{files_list}",
                    parse_mode="Markdown")
            await state.finish()
        else:
            await message.reply("Please enter a value between 0 and 1 with increments of 0.1")
    except ValueError:
        await message.reply("Please enter a valid number between 0 and 1 with increments of 0.1")


### /prompt
@dp.message_handler(commands=["prompt"])
async def setTemperature(message: types.Message):
    # Set state
    await Form.prompt.set()
    # Send a message to the user asking for the temperature
    await message.reply(
        f"Please reply with a prompt you would want me to use when generating responses\n\n✏️Current prompt: {user_prompt}\n\nWish to go back? Use /cancel.")


@dp.message_handler(state=Form.prompt)
async def setPrompt(message: types.Message, state: FSMContext):
    try:
        global user_prompt
        user_prompt = message.text

        if not files_uploaded:
            files_list = 'None'
            await message.reply(
                f"Prompt has been updated ✅\n\n*Current settings:*\n✏️Prompt: {user_prompt}\n\n🌡️Temperature: {temp_value}\n\n🔍Sources: {sources}\n\n📂Files uploaded: {files_list}",
                parse_mode="Markdown")
        else:
            files_list = "\n".join(files_uploaded)
            await message.reply(
                f"Prompt has been updated ✅\n\n*Current settings:*\n✏️Prompt: {user_prompt}\n\n🌡️Temperature: {temp_value}\n\n🔍Sources: {sources}\n\n📂Files uploaded:\n{files_list}",
                parse_mode="Markdown")
        await state.finish()
    except:
        await message.reply("Please enter a valid prompt")


### /sources
@dp.message_handler(commands=["sources"])
async def setTemperature(message: types.Message):
    # Set state
    await Form.n_sources.set()
    # Send a message to the user asking for the temperature
    await message.reply(
        "How many sources would you like me to look through when generating responses? 🔍\n\nWish to go back? Use /cancel.")


@dp.message_handler(state=Form.n_sources)
async def setPrompt(message: types.Message, state: FSMContext):
    global sources
    try:
        if float(message.text) in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            sources = int(message.text)
            if not files_uploaded:
                files_list = 'None'
                await message.reply(
                    f"Number of sources to look through has been updated ✅\n\n*Current settings:*\n✏️Prompt: {user_prompt}\n\n🌡️Temperature: {temp_value}\n\n🔍Sources: {sources}\n\n📂Files downloaded: {files_list}",
                    parse_mode="Markdown")
            else:
                files_list = "\n".join(files_uploaded)
                await message.reply(
                    f"Number of sources to look through has been updated ✅\n\n*Current settings:*\n✏️Prompt: {user_prompt}\n\n🌡️Temperature: {temp_value}\n\n🔍Sources: {sources}\n\n📂Files downloaded:\n{files_list}",
                    parse_mode="Markdown")

            await state.finish()
        else:
            await message.reply("Please enter an integer number between 0 and 10")
    except:
        await message.reply("Please enter an integer number between 0 and 10")


### /reset
@dp.message_handler(commands=["reset"])
async def reset(message: types.Message):
    global temp_value
    global user_prompt
    global sources
    temp_value = default_temp_value
    user_prompt = default_user_prompt
    sources = default_sources
    if not files_uploaded:
        files_list = 'None'
        await message.reply(
            f"Default settings restored ✅\n\n*Current settings:*\n🌡️Temperature: {temp_value}\n\n🔍Sources: {sources}\n\n📂Files uploaded: {files_list}",
            parse_mode="Markdown")
    else:
        files_list = "\n".join(files_uploaded)
        await message.reply(
            f"Default settings restored ✅\n\n*Current settings:*\n🌡️Temperature: {temp_value}\n\n🔍Sources: {sources}\n\n📂Files uploaded:\n{files_list}",
            parse_mode="Markdown")


### /settings
@dp.message_handler(commands=["settings"])
async def settings(message: types.Message):
    if not files_uploaded:
        files_list = 'None'
        await message.reply(
            f"*Current settings:*\n🌡️Temperature: {temp_value}\n\n🔍Sources: {sources}\n\n📂Files uploaded: {files_list}",
            parse_mode="Markdown")
    else:
        files_list = "\n".join(files_uploaded)
        await message.reply(
            f"*Current settings:*\n🌡️Temperature: {temp_value}\n\n🔍Sources: {sources}\n\n📂Files uploaded:\n{files_list}",
            parse_mode="Markdown")


### receiving documents
@dp.message_handler(content_types=types.ContentType.DOCUMENT)
async def handle_document(message: types.Message):
    loading_message = await message.answer(download_statement)
    await message.answer_chat_action("UPLOAD_DOCUMENT")
    # Check if the received document is a PDF
    if message.document.mime_type == 'application/pdf' or message.document.mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        # Get the file ID and file path
        file_id = message.document.file_id
        file_name = message.document.file_name
        file = await bot.get_file(file_id)
        file_path = file.file_path
        # Create the OnlinePDFLoader instance with the PDF URL
        if message.document.mime_type == 'application/pdf':
            loader = OnlinePDFLoader(f'https://api.telegram.org/file/bot{BOT_API_KEY}/{file_path}')
        elif message.document.mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            loader = Docx2txtLoader(f'https://api.telegram.org/file/bot{BOT_API_KEY}/{file_path}')
        # loader = DirectoryLoader("attachments/")
        data = loader.load()
        if all(d.page_content == '' for d in data):
            await message.reply(
                f"An unexpected error has occurred. Try resending your file in a Word/PDF format")
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                           chunk_overlap=200)
            texts = text_splitter.split_documents(data)

            docsearch = Pinecone.from_texts([t.page_content for t in texts],
                                            embed,
                                            index_name=PINECONE_INDEX_NAME,
                                            namespace=namespace_name)
            files_uploaded.append(file_name)
            files_list = "\n".join(files_uploaded)
            await message.reply(f"📂Files uploaded:\n{files_list}\n\nAsk me anything about the files!")

        await bot.delete_message(chat_id=loading_message.chat.id,
                                 message_id=loading_message.message_id)
    else:
        await message.reply(
            "Sorry, only PDF & word files are supported. Please try again!"
        )


###functions to be used later
button1 = InlineKeyboardButton(text="🔖 1", callback_data="s1")
button2 = InlineKeyboardButton(text="🔖 2", callback_data="s2")
button3 = InlineKeyboardButton(text="🔖 3", callback_data="s3")
button4 = InlineKeyboardButton(text="🔖 4", callback_data="s4")
button5 = InlineKeyboardButton(text="🔖 5", callback_data="s5")
button6 = InlineKeyboardButton(text="🔖 6", callback_data="s6")
button7 = InlineKeyboardButton(text="🔖 7", callback_data="s7")
button8 = InlineKeyboardButton(text="🔖 8", callback_data="s8")
button9 = InlineKeyboardButton(text="🔖 9", callback_data="s9")
button10 = InlineKeyboardButton(text="🔖 10", callback_data="s10")

keyboard_inline1 = InlineKeyboardMarkup(row_width=1).add(button1)
keyboard_inline2 = InlineKeyboardMarkup(row_width=2).add(button1, button2)
keyboard_inline3 = InlineKeyboardMarkup(row_width=3).add(button1, button2, button3)
keyboard_inline4 = InlineKeyboardMarkup(row_width=4).add(button1, button2, button3, button4)
keyboard_inline5 = InlineKeyboardMarkup(row_width=5).add(button1, button2, button3, button4, button5)
keyboard_inline6 = InlineKeyboardMarkup(row_width=4).add(button1, button2, button3, button4).add(button5, button6)
keyboard_inline7 = InlineKeyboardMarkup(row_width=4).add(button1, button2, button3, button4).add(button5, button6,
                                                                                                 button7)
keyboard_inline8 = InlineKeyboardMarkup(row_width=4).add(button1, button2, button3, button4).add(button5, button6,
                                                                                                 button7, button8)
keyboard_inline9 = InlineKeyboardMarkup(row_width=5).add(button1, button2, button3, button4, button5).add(button6,
                                                                                                          button7,
                                                                                                          button8,
                                                                                                          button9)
keyboard_inline10 = InlineKeyboardMarkup(row_width=5).add(button1, button2, button3, button4, button5).add(button6,
                                                                                                           button7,
                                                                                                           button8,
                                                                                                           button9,
                                                                                                           button10)


def cut_text_into_parts(text, max_length=4096):
    if len(text) <= max_length:
        return [text]

    parts = []
    current_part = ""
    lines = text.splitlines()

    for line in lines:
        if len(current_part) + len(line) + 1 <= max_length:  # +1 for the newline character
            current_part += line + "\n"
        else:
            parts.append(current_part.strip())
            current_part = line + "\n"

    if current_part:
        parts.append(current_part.strip())

    return parts


### receiving messages
@dp.message_handler()
async def reply(message: types.Message):
    loading_message = await message.answer(loading_statement)
    await message.answer_chat_action("typing")
    vectorstore = Pinecone(index,
                           embed.embed_query,
                           text_field,
                           namespace=namespace_name)
    global docs
    docs = []
    if sources > 0:
        docs = vectorstore.similarity_search(message.text, k=sources)

    prompt_message = f"""{user_prompt}

    Read the DOCUMENT delimited by '///' and then read the QUESTION BY THE HUMAN.
    Based on your knowledge from the document, create a final answer to the question by the human.

    DOCUMENT:
    ///
    {docs}
    ///


    QUESTION BY THE HUMAN: {message.text}
    Chatbot:"""

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "user", "content": message.text}],
                                            temperature=temp_value)
    message_parts = cut_text_into_parts(response["choices"][0]["message"]["content"])

    for item in message_parts:
        response_message = item

        if not docs:
            await message.reply(response_message)

        elif len(docs) == 1:
            keyboard = keyboard_inline1
            await message.reply(response_message, reply_markup=keyboard)


        elif len(docs) == 2:
            keyboard = keyboard_inline2
            await message.reply(response_message, reply_markup=keyboard)


        elif len(docs) == 3:
            keyboard = keyboard_inline3
            await message.reply(response_message, reply_markup=keyboard)

        elif len(docs) == 4:
            keyboard = keyboard_inline4
            await message.reply(response_message, reply_markup=keyboard)

        elif len(docs) == 5:
            keyboard = keyboard_inline5
            await message.reply(response_message, reply_markup=keyboard)

        elif len(docs) == 6:
            keyboard = keyboard_inline6
            await message.reply(response_message, reply_markup=keyboard)

        elif len(docs) == 7:
            keyboard = keyboard_inline7
            await message.reply(response_message, reply_markup=keyboard)

        elif len(docs) == 8:
            keyboard = keyboard_inline8
            await message.reply(response_message, reply_markup=keyboard)

        elif len(docs) == 9:
            keyboard = keyboard_inline9
            await message.reply(response_message, reply_markup=keyboard)

        elif len(docs) == 10:
            keyboard = keyboard_inline10
            await message.reply(response_message, reply_markup=keyboard)

    await bot.delete_message(chat_id=loading_message.chat.id,
                             message_id=loading_message.message_id)


# function for later
def remove_line_spacing(text):
    lines = text.splitlines()
    cleaned_text = " ".join(lines)
    return cleaned_text


###recieving the pressing of buttons
@dp.callback_query_handler(text=["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s10"])
async def buttonspressed(call: types.CallbackQuery):
    global prompt
    global chain
    if call.data == "s1":
        await call.message.answer(
            f'🔖 Source 1:\n\n{remove_line_spacing(docs[0].page_content)}')
    elif call.data == "s2":
        await call.message.answer(
            f'🔖 Source 2:\n\n{remove_line_spacing(docs[1].page_content)}')
    elif call.data == "s3":
        await call.message.answer(
            f'🔖 Source 3:\n\n{remove_line_spacing(docs[2].page_content)}')
    elif call.data == "s4":
        await call.message.answer(
            f'🔖 Source 4:\n\n{remove_line_spacing(docs[3].page_content)}')
    elif call.data == "s5":
        await call.message.answer(
            f'🔖 Source 5:\n\n{remove_line_spacing(docs[4].page_content)}')
    elif call.data == "s6":
        await call.message.answer(
            f'🔖 Source 6:\n\n{remove_line_spacing(docs[5].page_content)}')
    elif call.data == "s7":
        await call.message.answer(
            f'🔖 Source 7:\n\n{remove_line_spacing(docs[6].page_content)}')
    elif call.data == "s8":
        await call.message.answer(
            f'🔖 Source 8:\n\n{remove_line_spacing(docs[7].page_content)}')
    elif call.data == "s9":
        await call.message.answer(
            f'🔖 Source 9:\n\n{remove_line_spacing(docs[8].page_content)}')
    elif call.data == "s10":
        await call.message.answer(
            f'🔖 Source 10:\n\n{remove_line_spacing(docs[9].page_content)}')


if __name__ == "__main__":
    executor.start_polling(dp)
    ###
