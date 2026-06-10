# CONTRIBUTING.md Japanese Companion

Original English document: [`CONTRIBUTING.md`](../../CONTRIBUTING.md)

## English Reference

The original contributing guide explains fork/clone, branch creation, making changes, building and testing, committing, review workflow, and pre-commit formatting for upstream contribution.

> **日本語**
> 元の contribution guide は upstream に変更を送る場合の一般手順を説明します。この local Japanese overlay では、ユーザー指定により upstream review workflow は使わず、current branch で local commits と final push のみを行います。
>
> **学習メモ**
> 元文書の review workflow は原文の一部として残しますが、この作業では sample behavior を変えず、日本語 companion と `JP:` comment の品質確認に集中します。

## Build And Test Expectations

English anchor: contributors are expected to build and test so existing functionality is not broken.

> **日本語**
> CUDA sample の変更では、build が通ることと結果が正しいことの両方が重要です。ただし CUDA Toolkit、compiler、GPU、display stack、optional libraries がない環境では build/test を実行できません。その場合は skip 理由を記録します。
>
> **学習メモ**
> docs-only/comment-only の変更でも `git diff --check` と inventory script は実行します。build/test の成功は実行した場合だけ記録します。

## Formatting And Style

English anchor: formatting tools such as pre-commit and clang-format keep the repository consistent.

> **日本語**
> formatting は behavior ではなく保守性を守る工程です。この日本語化では English code、identifier、API、output strings を保持し、長い説明は companion docs に置きます。source 内の日本語は concise `JP:` comment に限定します。
>
> **学習メモ**
> CUDA code は小さな layout 変更でも review が難しくなるため、logic を変えない comment-only 変更でも diff を確認します。

## Commit Hygiene

English anchor: changes should be grouped into meaningful commits.

> **日本語**
> commit は bootstrap、inventory、source annotations、sample guides、theme/glossary/docs、verification のように cohesive unit で分けます。大きな自動生成差分でも目的が 1 つなら 1 commit にまとめ、status と generator を同じ commit に含めます。
>
> **学習メモ**
> `git diff --check` を commit 前に実行し、line ending warning と whitespace error を区別します。exit code が成功であることを確認します。

## Local Policy Notes

English anchor: original contribution guidance remains upstream-oriented; this overlay has stricter local rules.

> **日本語**
> この local work では upstream branch、main、master には push しません。original license、headers、attribution は変更しません。日本語 license explanation がある場合も unofficial reference として扱い、English license が authoritative です。
>
> **学習メモ**
> final report には branch、base/final commit、counts、commands run、skipped checks、blockers、upstream に変更を送っていない確認を残します。
