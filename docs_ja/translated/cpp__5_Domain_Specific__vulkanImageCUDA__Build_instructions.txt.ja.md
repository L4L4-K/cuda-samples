# cpp/5_Domain_Specific/vulkanImageCUDA/Build_instructions.txt Japanese Companion

Original English/source document: [`cpp/5_Domain_Specific/vulkanImageCUDA/Build_instructions.txt`](../../cpp/5_Domain_Specific/vulkanImageCUDA/Build_instructions.txt)

Role: Text build/run/reference document

## English Reference

Key lines:
- For Windows:
- Follow these steps once you have installed Vulkan SDK for Windows from https://www.lunarg.com/vulkan-sdk/
- -- Install GLFW3 library at suitable location
- -- Open the vulkanImageCUDA VS project file.
- To add the GLFW3 library path
- -- Right click on Project name "vulkanImageCUDA" click on "Properties"
- -- In Property pages window go to Linker -> General. Here in "Additional Libraries Directories" edit and add path to glfw3dll.lib
- To add the GLFW3 headers path

> **日本語**
> build/run/reference 用の text document です。command や expected value は英語を維持します。
>
> **学習メモ**
> 英語の command、API、target、expected output、license、attribution は翻訳せず、ここでは読む順序と注意点を補います。

## How To Read This With The Code

- この document が input、reference、design note、build note のどれかを確認します。
- 関連する source file と README を同じ directory から探します。
- binary document は変換せず、必要な時だけ原文 viewer で確認します。

> **日本語**
> document 単体で理解しようとせず、同じ directory の source、CMake、README、data/reference file と対応付けます。
>
> **学習メモ**
> build/run document は挙動を決めることがありますが、この companion は説明だけで build output や sample output を変更しません。

## Related Japanese Material

- [docs_ja/README.md](../README.md)
- [Build And Run Cheatsheet](../glossary/build_run.md)
- [Debugging, Profiling, And Testing](../themes/debugging_profiling_testing.md)
