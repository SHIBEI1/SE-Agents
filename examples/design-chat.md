# 即时聊天 MVP 设计规格

演示账号可登录，建立两人会话，发送文本并查询消息历史，使用 SQLite 保存数据。

| 编号 | 模块 | 约定 |
| --- | --- | --- |
| DS01 | 账号 | POST /sessions 校验身份。 |
| DS02 | 会话 | POST /conversations 创建两人会话，GET /conversations 返回自己的会话。 |
| DS03 | 消息 | POST /conversations/{id}/messages 校验成员身份并保存文本。 |
| DS04 | 历史 | GET /conversations/{id}/messages 按时间返回消息，非成员返回403。 |

会话成员甲发送消息后乙可查询。非成员不能发送或读取，未登录返回401。
首版仅含两人文本聊天，群聊、音视频、文件传输留待后续。
