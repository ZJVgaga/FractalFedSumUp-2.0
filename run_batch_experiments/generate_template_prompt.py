import os

# 定义要收集的文件列表及其路径
file_paths = {
    "client_template": "./run_one_experiment/run_and_get_results/client/client_template.py",
    "server_template": "./run_one_experiment/run_and_get_results/server/server_template.py",
    "run_and_get_result": "./run_one_experiment/run_and_get_results/run_and_get_results.py",
    "config": "./run_one_experiment/run_and_get_results/config.py",
    "init_basic_modules":"./run_one_experiment/run_and_get_results/init_basic_modules.py",
}

# 创建 template_prompt.txt 并写入内容
with open("template_prompt.txt", "w", encoding="utf-8") as outfile:
    for name, path in file_paths.items():
        outfile.write(f"===== {name} =====\n")
        
        try:
            with open(path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
        except FileNotFoundError:
            outfile.write(f"⚠️ 文件未找到: {path}\n")
        
        outfile.write("\n\n")
    
    # 添加最后的提示字符串
    prompt = """
    把下面的不规范的FL过程按照上面提供的标准过程规范，
    生成指定的类和函数（class Client,Server，init_basic_modules()），
    增加必要的参数设置（config,import_FL_modules(config)），
    所有的打印需要用logger完成
    要求你给出详尽的#中文注释，在适当的地方进行打印：下面是原始代码：
    """

    outfile.write(prompt)
print("template_prompt.txt 生成完成！")