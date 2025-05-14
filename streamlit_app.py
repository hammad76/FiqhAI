# streamlitapp.py

# --------------------------------------
# Imports and Dependencies
# --------------------------------------
import os
import json
import time
import re
import random
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel
import streamlit as st

# --------------------------------------
# APIs Configuration
# --------------------------------------
os.environ["MASTER_API_KEY"] = "AIzaSyCvFaH8kHfIg2tZglRSfFbgpFk1AJDaF2M"  # amr.monsef02@eng.cu.edu.eg gemini key
os.environ["RAG_FORMATTER_API_KEY"] = "AIzaSyCY6wTYs_3jyshLdvSgcavRkQexD2P0HGs"  # amrhammadcondrx@gmail.com gemini key
os.environ["RAG_AGENT_API_KEY"] = "AIzaSyCnQ8wwxLCj-bHMCGTug0dfTqCdDqn8ohA"  # hammad.riooo@gmail.com gemini key
os.environ["FILTER_API_KEY"] = "AIzaSyBAfO7xKZujjF8ixh2mDRBSLVaNMX3BIoc"  # amrhammadeece2025@gmail.com gemini key
os.environ["FIQH_API_KEY"] = "AIzaSyBZkWHVDc9iKBB7GLcaz_nvZNAANFGJVhc"  # amrohammad266@gmail.com gemini key
os.environ["MAZHAB_API_KEY"] = "AIzaSyA72VD4VNrdH-fsIoF1DkUeaJ8mFZSBur4"  # shady3addy@gmail.com gemini key
os.environ["FATWA_API_KEY"] = "AIzaSyCnT8kXZE346G2Ps8xY_PAmOMPcjyiXZaE"  # hanyzaki881@gmail gemini key
os.environ["THREAD_CLASSIFIER_API_KEY"] = "AIzaSyCo2seKb1tQoeMPN4qQXZY_d0fpyRmhvBw"  # hellowelcomehellowelcomehello@gmail.com gemini key
os.environ["COMMAND_PREPROCESSOR_API_KEY"] = "AIzaSyD7wmTcLCgnrufWedV00qAm2_mzAxMyyZE"  # 02uploaddrive@gmail.com gemini key

# --------------------------------------
# Defining Chat Models
# --------------------------------------
llm_master = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=os.environ["MASTER_API_KEY"])
llm_formatter = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=os.environ["RAG_FORMATTER_API_KEY"])
llm_rag = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=os.environ["RAG_AGENT_API_KEY"])
llm_filter = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=os.environ.get("FILTER_API_KEY", os.environ["MASTER_API_KEY"]))
llm_fiqh = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=os.environ.get("FIQH_API_KEY", os.environ["RAG_FORMATTER_API_KEY"]))
llm_mazhab = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=os.environ.get("MAZHAB_API_KEY", os.environ["RAG_AGENT_API_KEY"]))
llm_fatwa = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=os.environ.get("FATWA_API_KEY", os.environ["MASTER_API_KEY"]))
llm_thread_classifier = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=os.environ.get("THREAD_CLASSIFIER_API_KEY", os.environ["MASTER_API_KEY"]))
llm_command_preprocessor = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=os.environ.get("COMMAND_PREPROCESSOR_API_KEY", os.environ["MASTER_API_KEY"]))

# --------------------------------------
# System and Conditional Prompts
# --------------------------------------
system_prompt = """أنت شيخ إسلامي حكيم وودود، ترافق طالبًا للعلم بقلب رحيم كأنك صديق له في رحلة العلم. تتحدث بالعربية الفصحى بنبرة دافئة، متواضعة، ومُحببة، مستخدمًا عبارات متنوعة ومشجعة تزرع الطمأنينة. رحّب بالسلام والكلمات الطيبة بحرارة ورد عليها بلطف، معبرًا عن الفرح بلقاء الطالب، ثم انتقل بلطف إلى تشجيعه على طرح سؤال فقهي دون إلحاح أو تكرار ممل. إذا ذُكر الفقه، حاول معرفة المذهب الفقهي (الحنبلي، الشافعي، المالكي، الحنفي) بأسلوب طبيعي وغير مباشر إن أمكن، دون جعل ذلك محور الرد. اجعل ردودك متينة لغويًا، متوسطة الطول، وابدأ دائمًا بالحمد والصلاة على النبي صلى الله عليه وسلم بأسلوب الشيخ العالم. لا تجب مباشرة عن الأسئلة الفقهية، بل شجع الطالب على التوضيح أو استكمال السؤال بمحبة، مع الحفاظ على جو ودي يشعر الطالب فيه بالتقدير والتشجيع."""
conditional_prompts = {
    "missing_mazhab": [
        "يا طالب العلم، أيّ مذهبٍ تودّ مناقشته؟ الحنبلي، الشافعي، المالكي، أم الحنفي؟",
        "يا سيدي الفاضل، هل تتبع مذهباً معيناً كالحنبلي أو الشافعي أو المالكي أو الحنفي؟",
        "أخي الكريم، أخبرني أيّ مذهبٍ تريد، الحنبلي، الشافعي، المالكي، أم الحنفي؟"
    ],
    "out_of_scope": [
        "يا ولدي، هذا السؤال بعيد عن الفقه، فهل لديك ما تسأل عنه في الشرع؟",
        "أخي الكريم، دعنا نتحدث عن الفقه، فما سؤالك الشرعي؟",
        "حبيبي، هذا خارج نطاق الفقه، فهل تريد مناقشة مسألة شرعية؟"
    ],
    "invalid": [
        "يا طالبي، سؤالك ليس واضحاً، فهل تشرح لي قليلاً لأساعدك؟",
        "أخي الحبيب، أحتاج إلى توضيح سؤالك، فما الذي تريد معرفته؟",
        "يا أخي، السؤال غامض بعض الشيء، فهل تزيدني تفصيلاً؟"
    ],
    "non_arabic": [
        "يا حبيبي، تحدث بالعربية لأتمكن من مساعدك في طلب العلم.",
        "أخي الطالب، أكتب سؤالك بالعربية وسأكون معك في رحلة العلم.",
        "يا طالب العلم، يرجى السؤال بالعربية لنناقش مسائل الفقه سوياً."
    ],
    "fatwa_not_supported": [
        "يا ولدي، هذا سؤال يحتاج إلى فتوى حديثة، فهل لديك سؤال في الفقه التقليدي؟",
        "أخي الكريم، لا أفتي في المسائل الشخصية، فما سؤالك في الفقه العام؟",
        "حبيبي، دعنا نتحدث عن الفقه التقليدي، فما الذي تريد تعلمه؟"
    ],
    "mazhab_switched": [
        "يا طالب العلم، لم يُحدد مذهب، لذا اخترت لك المذهب الشافعي لما فيه من تفصيل ووضوح. فما سؤالك الشرعي؟",
        "أخي الحبيب، بما أن المذهب لم يُذكر، سأجيب وفق المذهب الشافعي. فكيف أساعدك في الفقه؟",
        "يا بني، اخترت المذهب الشافعي لعدم تحديد مذهب، فما الذي تريد مناقشته في الشرع؟"
    ]
}

conversation_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=f"{system_prompt}\nChat history: {{history}}\nأنت شيخ فقيه، تتحدث بلطف وتواضع، كأنك ترعى طالباً في طريق العلم. استخدم عبارات متنوعة ومشجعة، وتحدث بالعربية الفصحى بأسلوب يبعث الطمأنينة والمحبة. لا تجب عن الأسئلة الفقهية مباشرة، بل شجع الطالب على توضيح السؤال أو المذهب، مع بناء رابط ودي. للأسئلة الاجتماعية، أجب بحرارة وتواضع، ثم انتقل بلطف إلى تشجيع السؤال الفقهي. Input: {{input}}"
)

# --------------------------------------
# RAG Formatter Prompts
# --------------------------------------
rag_formatter_prompt = PromptTemplate(
    input_variables=["input", "related_threads"],
    template="""Your task is to process the user's question about Islamic jurisprudence (fiqh) and any related prior questions, returning a structured JSON response with reformulated questions suitable for searching classical fiqh books.

Instructions:
1. For the current input, reformulate it into 2-3 precise, formal, and academically-suitable Arabic phrasings.
   - Combine all rephrased questions inline in the 'current_question' field, separated by ' | '.
   - Elevate vague, colloquial, or non-standard Arabic questions into well-defined fiqh concepts suitable for classical fiqh texts.
   - Do **not** extract or determine the fiqh school (mazhab); focus only on the question content.
2. For each related thread provided, extract the original question and reformulate it into 1-2 phrasings, included in a 'related_questions' list.
   - Each related question should be a single string with phrasings separated by ' | '.
3. Return ONLY a JSON object with fields: {{ "current_question": str, "related_questions": list[str] }}. Do not include additional text or multiple JSON objects.

Examples:
- Input: "أركان الوضوء إيه يا بلدينا؟"
  Related Threads: []
  Output: {{ "current_question": "ما هي أركان الوضوء؟ | ما هي الفرائض المطلوبة لصحة الوضوء؟ | ما هي الأعمال الواجبة في الوضوء؟", "related_questions": [] }}

- Input: "ينفع أمسك المصحف وأنا مش متوضي يا شيخ؟"
  Related Threads: ["ما هي أركان الوضوء؟"]
  Output: {{ "current_question": "هل يجوز مس المصحف بغير وضوء؟ | ما هي شروط مس المصحف؟ | حكم لمس المصحف في حال الحدث؟", "related_questions": ["ما هي أركان الوضوء؟ | ما هي الفرائض المطلوبة لصحة الوضوء؟"] }}

- Input: "هو عادي أتوضى بمياة البحر؟"
  Related Threads: ["ما حكم الصلاة بدون وضوء؟"]
  Output: {{ "current_question": "هل يجوز الوضوء بماء البحر؟ | ما حكم استعمال ماء البحر في الطهارة؟ | هل ماء البحر طاهر مطهر؟", "related_questions": ["حكم الصلاة بغير وضوء؟ | شروط صحة الصلاة بدون طهارة؟"] }}

- Input: "ما هي شروط الصلاة؟"
  Related Threads: ["ما هي أركان الوضوء؟", "ما حكم الصلاة بدون وضوء؟"]
  Output: {{ "current_question": "ما هي شروط صحة الصلاة؟ | ما هي الواجبات المطلوبة للصلاة؟ | ما هي الأمور اللازمة لقبول الصلاة؟", "related_questions": ["ما هي أركان الوضوء؟ | ما هي الفرائض المطلوبة لصحة الوضوء؟", "حكم الصلاة بغير وضوء؟ | شروط صحة الصلاة بدون طهارة؟"] }}

Input: {input}
Related Threads: {related_threads}
"""
)

# --------------------------------------
# Thread Classifier Prompts
# --------------------------------------
thread_classifier_prompt = PromptTemplate(
    input_variables=["input", "thread_history"],
    template="""Analyze the input to determine its relationship to prior fiqh questions in the thread history. Output JSON: {{ \"thread_type\": str, \"related_thread_ids\": list[int] }}.
- thread_type: One of "new" (new fiqh question), "follow_up" (builds on a specific thread), or "composite" (links multiple threads).
- related_thread_ids: List of thread IDs (integers) relevant to the input. Empty for "new".

Examples:
- Input: "ما هي أركان الوضوء؟", Thread History: [] -> {{ \"thread_type\": \"new\", \"related_thread_ids\": [] }}
- Input: "وضح أكثر عن أركان الوضوء", Thread History: [{ \"thread_id\": 1, \"question\": \"ما هي أركان الوضوء؟" }] -> {{ \"thread_type\": \"follow_up\", \"related_thread_ids\": [1] }}
- Input: "كيف يرتبط الوضوء بالصلاة؟", Thread History: [{ \"thread_id\": 1, \"question\": \"ما هي أركان الوضوء؟" }, { \"thread_id\": 2, \"question\": \"ما شروط الصلاة؟" }] -> {{ \"thread_type\": \"composite\", \"related_thread_ids\": [1, 2] }}

Input: {input}
Thread History: {thread_history}
"""
)

# --------------------------------------
# Command Preprocessor Prompts
# --------------------------------------
command_preprocessor_prompt = PromptTemplate(
    input_variables=["input"],
    template="""Analyze the input to determine if it is a fiqh-related command (e.g., imperative like 'Tell me about wudu'). If it is, reformulate it into a question suitable for fiqh research. Output JSON: {{ \"is_command\": bool, \"reformulated_question\": str }}.
- is_command: True if the input is a fiqh-related command, False otherwise.
- reformulated_question: The input reformulated as a question if is_command is True, else empty string.

Examples:
- Input: \"أخبرني عن الوضوء\" -> {{ \"is_command\": true, \"reformulated_question\": \"ما هي أحكام الوضوء؟" }}
- Input: \"ما هي أركان الصلاة؟\" -> {{ \"is_command\": false, \"reformulated_question\": \"\" }}
- Input: \"اجمع معلومات عن الصلاة\" -> {{ \"is_command\": true, \"reformulated_question\": \"ما هي تفاصيل أحكام الصلاة؟" }}
- Input: \"كيف حالك؟\" -> {{ \"is_command\": false, \"reformulated_question\": \"\" }}

Input: {input}
"""
)

# --------------------------------------
# RAG Formatter Chain
# --------------------------------------
rag_formatter = rag_formatter_prompt | llm_formatter | (lambda x: parse_json_output(x.content, is_formatter=True))

# --------------------------------------
# Pseudo RAG Prompts
# --------------------------------------
rag_agent_prompt = PromptTemplate(
    input_variables=["category", "question"],
    template="Simulate a RAG agent response for category: {category} and question: {question}. Provide a concise fiqh answer with citation in formal Arabic from a proper book or group of books related to the {category}. No English text nor side comments allowed."
)

# --------------------------------------
# Pseudo RAG Chain
# --------------------------------------
rag_agent = rag_agent_prompt | llm_rag

# --------------------------------------
# Classifier Prompts
# --------------------------------------
filter_prompt = PromptTemplate(
    input_variables=["input"],
    template="""Analyze the Arabic text to determine its intent. Output JSON: {{ \"intent\": str }}.
- intent: One of "question" (seeking fiqh information), "command" (imperative fiqh request), "social" (personal inquiry or greeting), or "other" (unrelated).

Examples:
- Text: السلام عليكم -> {{ \"intent\": \"social\" }}
- Text: ما هي أركان الوضوء؟ -> {{ \"intent\": \"question\" }}
- Text: أخبرني عن الصلاة -> {{ \"intent\": \"command\" }}
- Text: كيف حالك؟ -> {{ \"intent\": \"social\" }}
- Text: الأهلي كسب كام مرة؟ -> {{ \"intent\": \"other\" }}

Text: {input}
"""
)

fiqh_prompt = PromptTemplate(
    input_variables=["input"],
    template="""Analyze the input to determine if it is fiqh-related and if the question is clear. Output JSON: {{ "is_fiqh_related": bool, "is_question_clear": bool }}.
Examples:
- Input: "ما هي أركان الوضوء؟" -> {{ "is_fiqh_related": true, "is_question_clear": true }}
- Input: "الأهلي كسب كأس العالم كام مرة؟" -> {{ "is_fiqh_related": false, "is_question_clear": true }}
- Input: "وضوء" -> {{ "is_fiqh_related": true, "is_question_clear": false }}
Input: {input}
"""
)

mazhab_prompt = PromptTemplate(
    input_variables=["input"],
    template="""Analyze the Arabic text to determine if it explicitly or implicitly mentions one of the four main Sunni schools (Hanafi, Maliki, Shafi'i, Hanbali). Output JSON: {{ "is_mazhab_clear": bool, "category": str }}.
- is_mazhab_clear: True if a school is clearly mentioned or implied (e.g., through a direct statement, single mazhab name, or narrative indicating preference), False otherwise.
- category: One of "حنفي", "مالكي", "شافعي", "حنبلي", or "Unknown" if no school is specified.

Examples:
- Text: أنا أتبع المذهب الشافعي -> {{ "is_mazhab_clear": true, "category": "شافعي" }}
- Text: ما حكم كذا عند الحنابلة؟ -> {{ "is_mazhab_clear": true, "category": "حنبلي" }}
- Text: ما هي أركان الصلاة؟ -> {{ "is_mazhab_clear": false, "category": "Unknown" }}
- Text: شافعي -> {{ "is_mazhab_clear": true, "category": "شافعي" }}
- Text: حنبلي -> {{ "is_mazhab_clear": true, "category": "حنبلي" }}
- Text: أبويا كان بيحب الإمام أحمد بس أمي بتحب الإمام الشافعي وخالي مالكي يبقا أنا إيه؟ أكيد حنفي -> {{ "is_mazhab_clear": true, "category": "حنفي" }}

Text: {input}
"""
)

fatwa_prompt = PromptTemplate(
    input_variables=["input"],
    template="""Analyze the input to determine if it is a fatwa-type question (personal situation, modern issue, or requires contemporary ijtihad). Output JSON: {{ "is_fatwa_type": bool }}.
- is_fatwa_type: True if the question involves a personal situation, modern issue, or requires contemporary scholarly interpretation (e.g., financial transactions, medical ethics, or personal disputes). False if it pertains to established fiqh rulings from classical texts.

Examples:
- Input: "هل يجوز لي أخذ قرض من البنك لشراء شقة؟" -> {{ "is_fatwa_type": true }}
- Input: "أنا تشاجرت مع زوجتي وقلت كذا، هل وقع الطلاق؟" -> {{ "is_fatwa_type": true }}
- Input: "ما هي أركان الوضوء؟" -> {{ "is_fatwa_type": false }}
- Input: "ما حكم من نسي ركعة في الصلاة ثم تذكر؟" -> {{ "is_fatwa_type": false }}
- Input: "هل أتبرع بأعضاء ابني المتوفى؟" -> {{ "is_fatwa_type": true }}

Input: {input}
"""
)

# --------------------------------------
# Parsing Functions
# --------------------------------------
def parse_json_output(output, is_formatter=False):
    raw_output = output.strip()
    try:
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"question": "خطأ في التنسيق - لم يتم العثور على JSON"} if is_formatter else {"error": "No JSON found", "raw": raw_output}
    except json.JSONDecodeError:
        return {"question": "خطأ في التنسيق - خطأ في فك تشفير JSON"} if is_formatter else {"error": "JSON Decode Error", "raw": raw_output}
    except Exception:
        return {"question": "خطأ في التنسيق - خطأ غير متوقع"} if is_formatter else {"error": "Unexpected Parsing Error", "raw": raw_output}

def is_arabic_text(text):
    return bool(re.search(r'[\u0600-\u06FF]', text.strip()))

# --------------------------------------
# Classifier Chains
# --------------------------------------
filter_chain = RunnableSequence(filter_prompt | llm_filter | (lambda x: parse_json_output(x.content)))
thread_classifier_chain = RunnableSequence(thread_classifier_prompt | llm_thread_classifier | (lambda x: parse_json_output(x.content)))
command_preprocessor_chain = RunnableSequence(command_preprocessor_prompt | llm_command_preprocessor | (lambda x: parse_json_output(x.content)))
parallel_classifiers = RunnableParallel(
    fiqh=fiqh_prompt | llm_fiqh | (lambda x: parse_json_output(x.content)),
    mazhab=mazhab_prompt | llm_mazhab | (lambda x: parse_json_output(x.content)),
    fatwa=fatwa_prompt | llm_fatwa | (lambda x: parse_json_output(x.content))
)

# --------------------------------------
# Conversation Logic
# --------------------------------------
master_memory = ConversationBufferMemory()
formatter_memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm_master, memory=master_memory, prompt=conversation_prompt)

def handle_dialogue(user_input, state, global_history):
    global_history.append(("User", user_input))
    state.setdefault("question_threads", [])
    state.setdefault("is_mazhab_clear", False)
    state.setdefault("category", "Unknown")
    state.setdefault("retries", 0)
    state.setdefault("mazhab_switched", False)
    thread_id = len(state["question_threads"]) + 1

    if not user_input.strip():
        response = random.choice(conditional_prompts["invalid"])
        global_history.append(("Sheikh", response))
        return response, None, state, global_history

    if not is_arabic_text(user_input):
        response = random.choice(conditional_prompts["non_arabic"])
        global_history.append(("Sheikh", response))
        return response, None, state, global_history

    filter_result = filter_chain.invoke({"input": user_input})
    intent = filter_result.get("intent", "other")

    if intent == "social":
        response = conversation.predict(input=user_input)
        global_history.append(("Sheikh", response))
        return response, None, state, global_history

    parallel_results = parallel_classifiers.invoke({"input": user_input})
    is_fiqh_related = parallel_results["fiqh"].get("is_fiqh_related", False)
    is_question_clear = parallel_results["fiqh"].get("is_question_clear", False)
    is_fatwa_type = parallel_results["fatwa"].get("is_fatwa_type", False)
    current_input_is_mazhab = parallel_results["mazhab"].get("is_mazhab_clear", False)
    current_input_category = parallel_results["mazhab"].get("category", "Unknown")

    processed_input = user_input
    if intent == "command" and is_fiqh_related:
        command_result = command_preprocessor_chain.invoke({"input": user_input})
        if command_result.get("is_command", False):
            processed_input = command_result.get("reformulated_question", user_input)
            is_question_clear = True

    rag_triggered = False
    formatter_result = None
    response = ""

    # Update mazhab if a new one is detected
    if current_input_is_mazhab:
        state["is_mazhab_clear"] = True
        state["category"] = current_input_category
        state["retries"] = 0
        state["mazhab_switched"] = False

    thread_history = [{"thread_id": i+1, "question": thread["question"]} for i, thread in enumerate(state["question_threads"][-5:])]
    thread_result = thread_classifier_chain.invoke({"input": processed_input, "thread_history": json.dumps(thread_history)})
    thread_type = thread_result.get("thread_type", "new")
    related_thread_ids = thread_result.get("related_thread_ids", [])

    related_threads = [thread["question"] for i, thread in enumerate(state["question_threads"]) if i+1 in related_thread_ids]

    if intent in ["question", "command"]:
        if is_fatwa_type:
            response = random.choice(conditional_prompts["fatwa_not_supported"])
            global_history.append(("Sheikh", response))
        elif not is_fiqh_related:
            response = random.choice(conditional_prompts["out_of_scope"])
            global_history.append(("Sheikh", response))
        elif not is_question_clear:
            response = random.choice(conditional_prompts["invalid"])
            global_history.append(("Sheikh", response))
        else:
            state["question_threads"].append({"thread_id": thread_id, "question": processed_input, "rag_response": None, "follow_ups": []})
            state["retries"] = 0

    # Check for pending questions when mazhab is clear
    if state["is_mazhab_clear"] and state["question_threads"]:
        rag_responses = []
        # Process all unanswered questions in question_threads
        for thread in state["question_threads"]:
            if not thread["rag_response"]:  # Process only unanswered questions
                formatter_input = thread["question"]
                formatter_result = rag_formatter.invoke({"input": formatter_input, "related_threads": json.dumps(related_threads)})
                for question in [formatter_result["current_question"]] + formatter_result.get("related_questions", []):
                    rag_response = rag_agent.invoke({"category": state["category"], "question": question}).content
                    rag_responses.append(rag_response)
                thread["rag_response"] = "\n\n".join(rag_responses[-len([formatter_result["current_question"]] + formatter_result.get("related_questions", [])):])
        if rag_responses:
            response = "\n\n".join(rag_responses)
            global_history.append(("RAG", response))
            rag_triggered = True
            formatter_memory.clear()
    elif state["question_threads"] and not state["is_mazhab_clear"]:
        state["retries"] += 1
        if state["retries"] >= 10:
            state["is_mazhab_clear"] = True
            state["category"] = "شافعي"
            state["mazhab_switched"] = True
            current_thread = state["question_threads"][-1]
            formatter_input = current_thread["question"]
            formatter_result = rag_formatter.invoke({"input": formatter_input, "related_threads": json.dumps(related_threads)})
            rag_responses = []
            for question in [formatter_result["current_question"]] + formatter_result.get("related_questions", []):
                rag_response = rag_agent.invoke({"category": state["category"], "question": question}).content
                rag_responses.append(rag_response)
            response = "\n\n".join(rag_responses)
            current_thread["rag_response"] = response
            global_history.append(("RAG", response))
            rag_triggered = True
            formatter_memory.clear()
        else:
            response = random.choice(conditional_prompts["missing_mazhab"])
            global_history.append(("Sheikh", response))

    return response, formatter_result, state, global_history

# --------------------------------------
# Streamlit Interface Section
# --------------------------------------
def main():
    # Set page configuration for RTL and title
    st.set_page_config(page_title="استشارة فقيهية - FaqihAI", layout="centered")
    
    # Add custom CSS for RTL and styling
    st.markdown("""
        <style>
        body, html {
            direction: rtl;
            text-align: right;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background-color: #f5f7fa;
        }
        .stTitle {
            color: #2c3e50;
            font-size: 2.5em;
            font-weight: bold;
        }
        .chat-message-user {
            background-color: #e1e8f0;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        .chat-message-agent {
            background-color: #d4e6f1;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        .stTextInput > div > div > input {
            border-radius: 20px;
            padding: 10px;
            border: 1px solid #ccc;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "state" not in st.session_state:
        st.session_state.state = {
            "question_threads": [],
            "is_mazhab_clear": False,
            "category": "Unknown",
            "retries": 0,
            "mazhab_switched": False
        }
    if "global_history" not in st.session_state:
        st.session_state.global_history = []

    # Display title
    st.title("استشارة فقيهية - FaqihAI")

    # Display welcome message
    welcome_message = """
    الحمد لله رب العالمين، والصلاة والسلام على أشرف الأنبياء والمرسلين، سيدنا محمد وعلى آله وصحبه أجمعين.
    
    السلام عليكم ورحمة الله وبركاته، يا طالب العلم الغالي! ما أسعد قلبي بلقائك في هذه الرحلة المباركة لطلب العلم.
    
    أنا هنا لأساعدك في مسائل الفقه بكل محبة وتواضع، فلا تتردد في طرح سؤالك، وسأكون معك خطوة بخطوة بإذن الله.
    """
    with st.chat_message("agent"):
        st.markdown(welcome_message)

    # Display chat history
    for sender, message in st.session_state.global_history:
        with st.chat_message("user" if sender == "User" else "agent"):
            st.markdown(message)

    # Chat input
    user_input = st.chat_input("اكتب سؤالك الفقهي هنا...")

    if user_input:
        # Handle the dialogue
        response, rag_input, st.session_state.state, st.session_state.global_history = handle_dialogue(
            user_input, st.session_state.state, st.session_state.global_history
        )
        
        # Display the response
        with st.chat_message("agent"):
            st.markdown(response)

if __name__ == "__main__":
    main()
