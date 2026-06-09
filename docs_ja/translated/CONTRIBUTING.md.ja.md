# CONTRIBUTING.md Japanese Companion

Original English document: [`CONTRIBUTING.md`](../../CONTRIBUTING.md)

## English Reference

The original contributing guide explains fork/clone, branch creation, making changes, building and testing, committing, pull requests, and pre-commit formatting.

> **日本語**
> 元の contribution guide は、fork/clone、branch 作成、変更、build/test、commit、pull request、pre-commit による formatting を説明しています。
>
> **学習メモ**
> このローカル日本語化作業では、ユーザー指定により PR を作成しません。元文書の PR 手順は upstream contribution 用の一般説明として残します。

## Build And Test

English reference: contributors should build and run tests so changes do not break existing functionality.

> **日本語**
> CUDA sample の変更では、build が通ることと実行結果が正しいことの両方を確認します。環境に GPU/toolkit/display dependency がない場合は、skip した理由を記録します。
>
> **学習メモ**
> comment-only や docs-only の変更でも、`git diff --check` で whitespace と patch hygiene を確認します。

## Formatting

English reference: pre-commit hooks run linters and formatters, including `clang-format` for C++ and CUDA code.

> **日本語**
> formatting は挙動ではなく保守性を守るための工程です。この日本語化では英語 code、identifier、API、出力文字列を維持し、必要な日本語は `JP:` comment と companion docs に限定します。
>
> **学習メモ**
> 既存 code style を壊さないため、logic の周辺に長い翻訳文を埋め込まず、短い学習 comment にします。
