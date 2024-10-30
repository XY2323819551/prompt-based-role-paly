import asyncio
import sys
from pathlib import Path
from typing import List, Dict

# 将项目根目录添加到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from llm_pool.llm import get_model_response



async def self_refine_translation(source_text: str) -> str:

    # 第一次翻译
    first_translate_system_message = "You are an expert linguist, specializing in translation from English to Chinese."
    first_user_translation_prompt = f"""This is an English to Chinese translation, please provide the Chinese translation for this text. \
Do not provide any explanations or text apart from the translation.
English: {source_text}

Chinese:"""

    message1 = [
        {"role": "system", "content": first_translate_system_message},
        {"role": "user", "content": first_user_translation_prompt}
    ]

    translation_1 = await get_model_response(model_name="deepseek-chat", messages=message1)

    # 第二次反馈
    reflect_system_prompt = """You are an expert linguist specializing in translation from English to Chinese. \
You will be provided with a source text and its translation and your goal is to improve the translation."""

    reflect_prompt = f"""Your task is to carefully read a source text and a translation from English to Chinese, and then give constructive criticism and helpful suggestions to improve the translation. \

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's 
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying Chinese grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms Chinese).

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    message2 = [
        {"role": "system", "content": reflect_system_prompt},
        {"role": "user", "content": reflect_prompt}
    ]

    reflection = await get_model_response(model_name="deepseek-chat", messages=message2)

    # 第三次优化翻译
    refiner_system_prompt = "You are an expert linguist, specializing in translation editing from English to Chinese."

    refined_translate_user_prompt = f"""Your task is to carefully read, then edit, a translation from English to Chinese, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying Chinese grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else."""

    message3 = [
        {"role": "system", "content": refiner_system_prompt},
        {"role": "user", "content": refined_translate_user_prompt}
    ]

    final_translation = await get_model_response(model_name="deepseek-chat", messages=message3)
    return final_translation

async def main():
    source_text = """
    Last week, I spoke about AI and regulation at the U.S. Capitol at an event that was attended by legislative and business leaders. I'm encouraged by the progress the open source community has made fending off regulations that would have stifled innovation. But opponents of open source are continuing to shift their arguments, with the latest worries centering on open source's impact on national security. I hope we'll all keep protecting open source!

    Based on my conversations with legislators, I'm encouraged by the progress the U.S. federal government has made getting a realistic grasp of AI's risks. To be clear, guardrails are needed. But they should be applied to AI applications, not to general-purpose AI technology.
    """
    result = await self_refine_translation(source_text)
    print("Final translation:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
