Refactor @backend/ai_generator.py to support sequential tool calling where Claude can
make up to 2 tool calls in seperate API rounds.

Current behaviour:
- Claude make 1 tool call -> tools are removed from API params -> final response
- If Claude wants another tool call after seeing results, it can't (gets empty response)

Desired behavior:
- Each tool call should be a seperate API request where Claude can reason about previous results
- Support for complex queries requiring multiple searches for comparisions, multi-part questions, or
when information from different courses/lessons is needed

Example flow:
1. User: "Search for a course that discusses the same topic as lesson 4 of course X"
2. Claude: get course outline for course X -> hets title of lesson 4
3. Claude: uses the title to search for a course that discusses the same topic -> returns course information
4. Claude: provides complete answer

Requirements:
- Maximun 2 sequential rounds per user query
- Terminate when: (a) 2 rounds completed, (b) Claude's response has no tool_use blocks, or (c) tool call fails

Notes:
- update the system prompt in @backend/ai_generator.py
- update the test @backend/tests/tes_ai_generator.py
- Write tests that verify the external behavior (API calls made, tools executed, results returned) rather than internal state details.


Use two parallel subagents to brainstorm possible plans. Do not implement any code.