# 项目级 Codex skills

安装目录为仓库根目录的 `.agents/skills/`。已有 `tutor/` 保持不变。
本次只新增以下四个技能的完整目录，未安装整个 Superpowers 插件：

- `brainstorming/`
- `writing-plans/`
- `test-driven-development/`
- `requesting-code-review/`

## 来源与安装

来源：https://github.com/obra/superpowers

固定上游提交：`b36e0829c6d0140e93cfef2ca599b1b07d4a7797`。
四个目录内的文件内容保持上游原样，附带资源一并保留。
MIT 许可证见本目录的 `LICENSE.superpowers`。
下载器不保留 ZIP 中的执行权限，因此恢复了上游两个 `.sh` 脚本的可执行位。

安装使用 Codex 自带 skill-installer 的 `install-skill-from-github.py`，参数为：

```text
--repo obra/superpowers
--ref b36e0829c6d0140e93cfef2ca599b1b07d4a7797
--path skills/brainstorming skills/writing-plans skills/test-driven-development skills/requesting-code-review
--dest /home/mengjun/AMAT5315-2026Fall-Exercise/.agents/skills
```

重新克隆本课程仓库即可取得这些技能，无须再安装；安装器会拒绝覆盖同名目录。
无需添加 `config.toml`、启动钩子、全局符号链接或插件配置。

## 依赖与使用边界

- **brainstorming**：后续设计完成时调用 `writing-plans`，已包含在四项中。
  附带 `spec-document-reviewer-prompt.md`、`visual-companion.md` 和完整 `scripts/`。
  可选浏览器 companion 需要 Bash、Node.js 和浏览器；脚本仅使用 Node.js 内置模块，
  无 npm 包需要安装。本次环境未找到 `node`，未安装或启动此可选功能；文字讨论不依赖它。
  上游可选的 `elements-of-style:writing-clearly-and-concisely` 未安装。
- **writing-plans**：附带 `plan-document-reviewer-prompt.md`。
  上游要求在执行计划时使用 `superpowers:subagent-driven-development`
  或 `superpowers:executing-plans`，隔离 worktree 路径还引用
  `superpowers:using-git-worktrees`。这些属于后续执行阶段，超出本次四项安装范围，
  未安装；不可声称完整执行链已就绪。将来选择这些流程时再处理相应依赖。
- **test-driven-development**：必读的 `writing-good-tests.md` 已随目录安装。
  文档中的 npm/TypeScript 命令是示例，不是本 Rust 项目的安装依赖；
  `superpowers:writing-skills` 是关于技能行为测试的背景引用，非本次运行依赖。
- **requesting-code-review**：`code-reviewer.md` 是必须随附的审查提示模板，已安装。
  当前上游使用通用子代理，不要求另外安装名为 `code-reviewer` 的技能或插件代理。
  实际运行需要所在 Codex 会话支持子代理；本次仅安装，未启动审查子代理。

这些文件按原样保存，不把上游的 `superpowers:` 流程引用伪装成已安装的插件命令。
本地四个技能按其 `name` 调用，例如 `$writing-plans`。

## 新会话确认加载

在本仓库根目录或其子目录中开启新的 Codex 会话。Codex 会从当前目录向上到仓库根目录
扫描 `.agents/skills/`。CLI / IDE 中可用 `/skills` 或输入 `$` 查看技能选择列表，
确认以上四个名字存在（以及原来的 `tutor`）。

也可以发送以下核对请求，不启动设计工作：

> 请仅根据当前会话的可用技能列表，确认 brainstorming、writing-plans、
> test-driven-development、requesting-code-review 是否已被发现，并列出各自的
> SKILL.md 路径。不要调用这些技能，不要开始 Part 3。

技能被发现时会加载名称、描述和路径；完整指令在实际使用时才读取。
Codex 通常会自动检测新增技能，下一轮可用；若没有显示，重启 Codex 后再次检查。
文件安装检查不等同于已验证另一个新会话的加载结果。

官方说明：https://learn.chatgpt.com/docs/build-skills

本次只完成安装准备，没有开始 Part 3 的设计或实现。
