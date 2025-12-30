# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_009/main.py
import streamlit as st
from langchain.agents import create_agent as create_langchain_agent
from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.messages import HumanMessage

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# custom tools
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page

###### dotenv を利用しない場合は消してください ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################

CUSTOM_SYSTEM_PROMPT = """
あなたは、ユーザーのリクエストに基づいてインターネットで調べ物を行うアシスタントです。
利用可能なツールを使用して、調査した情報を説明してください。
既に知っていることだけに基づいて答えないでください。回答する前にできる限り検索を行ってください。
(ユーザーが読むページを指定するなど、特別な場合は、検索する必要はありません。)

検索結果ページを見ただけでは情報があまりないと思われる場合は、次の2つのオプションを検討して試してみてください。

- 検索結果のリンクをクリックして、各ページのコンテンツにアクセスし、読んでみてください。
- 1ページが長すぎる場合は、3回以上ページ送りしないでください（メモリの負荷がかかるため）。
- 検索クエリを変更して、新しい検索を実行してください。
- 検索する内容に応じて検索に利用する言語を適切に変更してください。
  - 例えば、プログラミング関連の質問については英語で検索するのがいいでしょう。

ユーザーは非常に忙しく、あなたほど自由ではありません。
そのため、ユーザーの労力を節約するために、直接的な回答を提供してください。

=== 悪い回答の例 ===
- これらのページを参照してください。
- これらのページを参照してコードを書くことができます。
- 次のページが役立つでしょう。

=== 良い回答の例 ===
- これはサンプルコードです。 -- サンプルコードをここに --
- あなたの質問の答えは -- 回答をここに --

回答の最後には、参照したページのURLを**必ず**記載してください。（これにより、ユーザーは回答を検証することができます）

ユーザーが使用している言語で回答するようにしてください。
ユーザーが日本語で質問した場合は、日本語で回答してください。ユーザーがスペイン語で質問した場合は、スペイン語で回答してください。
"""


def init_page():
    st.set_page_config(
        page_title="Web Browsing Agent",
        page_icon="🤗"
    )
    st.header("Web Browsing Agent 🤗")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.messages = [
            {"role": "assistant", "content": "こんにちは！なんでも質問をどうぞ！"}
        ]


def select_model():
    models = ("GPT-4", "Claude 4.5 Sonnet", "Gemini 2.5 Flash", "GPT-3.5 (not recommended)")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-3.5 (not recommended)":
        return ChatOpenAI(
            temperature=0, model_name="gpt-3.5-turbo")
    elif model == "GPT-4":
        return ChatOpenAI(
            temperature=0, model_name="gpt-4o")
    elif model == "Claude 4.5 Sonnet":
        return ChatAnthropic(
            temperature=0, model_name="claude-sonnet-4-5-20250929")
    elif model == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(
            temperature=0, model="gemini-2.5-flash")


def create_agent():
    tools = [search_ddg, fetch_page]
    llm = select_model()

    # LangChainのcreate_agentを使用（システムプロンプトを含める）
    agent = create_langchain_agent(
        llm,
        tools,
        system_prompt=CUSTOM_SYSTEM_PROMPT
    )
    return agent


def main():
    init_page()
    init_messages()
    web_browsing_agent = create_agent()

    # チャット履歴を表示
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="2023 FIFA 女子ワールドカップの優勝国は？"):
        # ユーザーメッセージを表示
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # コールバック関数の設定 (エージェントの動作の可視化用)
            st_cb = StreamlitCallbackHandler(
                st.container(), expand_new_thoughts=True)

            # エージェントを実行
            response = web_browsing_agent.invoke(
                {"messages": st.session_state.chat_history + [HumanMessage(content=prompt)]},
                config=RunnableConfig({'callbacks': [st_cb]})
            )

            # 最後のメッセージを取得
            last_message = response["messages"][-1]
            answer = last_message.content

            # チャット履歴に追加
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(last_message)

            # 表示用メッセージに追加
            st.session_state.messages.append({"role": "assistant", "content": answer})

            st.write(answer)


if __name__ == '__main__':
    main()
