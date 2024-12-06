import os
import json
import shutil
import asyncio
from zhipuai import ZhipuAI
from llm_pool.llm import get_model_response_stream, get_model_response


# tool1 获取文件
async def get_files(dir_path):
    files = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            files.append(file_path)
        elif os.path.isdir(file_path):
            files.extend(await get_files(file_path))
    return "文件夹下的文件为：" + "\n".join(files)

async def get_dir_files(dir_path="", params_format=False):
    if params_format:
        return ['dir_path']
    else:
        return await get_files(dir_path)



# tool2 创建新的文件夹
async def mkdir_dir(dir_tree=""):
    print("dir_tree:", dir_tree)
    all_need_dirs = dir_tree.replace("\\\\", "\\").split("\n")
    for dir in all_need_dirs:
        os.makedirs(dir, exist_ok=True)
    return "dir_tree: “{dir_tree}” have make success!"

async def mkdir_dirs(dir_tree="", params_format=False):
    if params_format:
        return ['dir_tree']
    else:
        return await mkdir_dir(dir_tree)



# tool3 移动文件
async def move_files(map={}, sources_dirs="", target_dirs="", params_format=False):
    if params_format:
        return ['map', 'sources_dirs', 'target_dirs']
    else:
        for file_name, category in map.items():
            source_file_path = sources_dirs + "\\" + file_name
            source_file_path = source_file_path.replace("\\\\","\\").replace("/", "\\")
            if os.path.exists(source_file_path):
                target_full_dir = target_dirs + "\\" + category
                target_full_dir = target_full_dir.replace("\\\\","\\").replace("/", "\\")
                shutil.move(source_file_path, target_full_dir)
            else:
                print(f"File {source_file_path} does not exist.")
    return "files have been moved!"



class MapperAgent:
    def __init__(self, model="deepseek-chat"):
        self.client = ZhipuAI(api_key="352894d8c48fd2e0b0547b3159cca22a.ZJ84NRbCzMiDaxMa")
        self.model = "glm-4-plus"
        self.tools = {"get_dir_files": get_dir_files, "mkdir_dirs": mkdir_dirs, "move_files": move_files}
        self.role = """
# Role: 文件整理员

# Profiles: 
- describe: 你是一个文件整理人员，负责将一个文件夹下的所有文件分类后，按类别整理到一个指定目录下，你可以使用工具完成一些步骤，Tools下的列出的工具就是你可以使用的全部工具。

# Goals:
- 你需要帮助其他人做文件整理，将一个文件夹下的所有文件自动分类，然后把每个文件整理到对应的类别文件夹之下。

# Tools:
- get_dir_files: 读取一个文件夹下所有的文件，返回文件名和文件路径，需要参数{"dir_path":"xxxx"}。
- mkdir_dirs: 根据给定的目录树创建对应的文件夹，，需要参数{"dir_tree":"xxxx"}。
- move_files: 将文件移动到指定的目录下，需要参数{"map":{"file1":"类别1", "file2":"类别2"}, "sources_dirs":input_dir, "target_dirs":output_dir}。

# Constraints:
- 你设计的类别一定是根据文件的特点和内容来合理设计的，不能随意分类。
- 你在设计类别的时候要覆盖全部的文件。
- 在把文件整理到对应的类别目录下的时候，不能有文件被遗漏。
- 整理后文件夹的目录根源目录在同一级目录下。
- 你在调用工具的时候可以使用“=>$tool_name: {key:value}”来触发工具调用。
- 每一次触发了不同的tool之后，你需要停止作答，等待用户调用对应的tool处理之后，将tool的结果重新组织语言后再继续作答，新的答案要接着“=>$tool_name”前面的最后一个字符继续生成结果，要保持结果通顺。

# Workflows:
- 1. 输入一个文件夹路径，获取该文件夹下的所有文件名。
- 2. 根据文件的特点和内容，设计合理的类别，类别输出的标识符为=>['类别1','类别2']。
- 3. 按照设计的类别新建类别文件夹，并输出文件夹的目录树，并创建好目录结构。
- 4. 将每个文件按照设计好的类别分类输出结果为=>{"file1":"类别1", "file2":"类别2"}。
- 5. 将每个文件按照分类结果移动到对应的类别文件夹下。

需要整理的文件夹目录为:{input_dir},整理后的文件夹目录为:{output_dir}。
"""

    async def tool_run(self, tool_message):
        function_name, function_params = tool_message.split(":", 1)
        function_params_json = json.loads(function_params)
        need_params = await self.tools[function_name](params_format=True)
        extract_params = {}
        for param in need_params:
            extract_params[param] = function_params_json[param]
        result = await self.tools[function_name](**extract_params)
        return result

    async def do_mapper(self, input_dir, output_dir, processed_content=""):
        # TODO: Implement the mapper logic
        prompt = self.role.replace("{input_dir}", input_dir).replace("{output_dir}", output_dir).strip()
        if len(processed_content) > 0:
            prompt = prompt + "\n\n" + processed_content
        messages = [{"role": "user", "content": prompt}]
        
        result = await get_model_response_stream(model_name="deepseek-chat", messages=messages)  # deepseek-chat、gpt-4o
        result = await get_model_response_stream(model_name="gpt-4o", messages=messages)  # deepseek-chat、gpt-4o
    
        # result = self.client.chat.completions.create(
        #         model=self.model,  # 请填写您要调用的模型名称
        #         messages=messages,
        #         temperature=0.0,
        #         stream=True
        #     )
    
        all_answer = ""
        tool_messages = ""
        tool_Flag = False

        async for chunk in result:
            all_answer += chunk.choices[0].delta.content
            if tool_Flag:
                tool_messages += chunk.choices[0].delta.content
                print(f"--------tool_messages--------:{tool_messages}")
                continue
            if ":" in chunk.choices[0].delta.content and ("=>$" in all_answer or "=>" in all_answer):
                breakpoint()
                tool_Flag = True
                tool_messages += chunk.choices[0].delta.content
                yield ": "
                continue
            yield chunk.choices[0].delta.content
        
        if tool_Flag:
            # breakpoint()
            tool_messages = all_answer.split("=>$")[-1] if "=>$" in all_answer else all_answer.split("=>")[-1]
            result = await self.tool_run(tool_message=tool_messages)
            for item in str(result+"\n"):
                yield item
            processed_content = processed_content + "\n" + "已经执行内容:" + all_answer + "\n" + "工具执行结果:" + result
            async for item in self.do_mapper(input_dir, output_dir, processed_content):
                yield item




input_dir = "/Users/zhangxiaoyu/Downloads/llm_arch_paper_test"
output_dir = "/Users/zhangxiaoyu/Downloads/paper_test"

mapper_agent = MapperAgent(model="deepseek-chat")
async def mapper():
    async for item in mapper_agent.do_mapper(input_dir, output_dir):
        yield item

async def main():
    async for item in mapper():
        print(item, end="", flush=True)
    return


if __name__ == '__main__':
    asyncio.run(main())
