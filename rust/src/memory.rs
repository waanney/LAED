use std::collections::VecDeque;

#[derive(Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Clone)]
pub enum ChatRole {
    User,
    Assistant,
}

pub struct ConversationMemory {
    messages: VecDeque<ChatMessage>,
    max_exchanges: usize,
    max_characters: usize,
}

impl ConversationMemory {
    pub fn new(max_exchanges: usize, max_characters: usize) -> Self {
        Self {
            messages: VecDeque::new(),
            max_exchanges,
            max_characters,
        }
    }

    pub fn snapshot(&self) -> Vec<ChatMessage> {
        self.messages.iter().cloned().collect()
    }

    pub fn add_exchange(&mut self, user: String, assistant: String) {
        self.messages.push_back(ChatMessage { role: ChatRole::User, content: user });
        self.messages.push_back(ChatMessage { role: ChatRole::Assistant, content: assistant });
        while self.messages.len() > self.max_exchanges * 2
            || self.messages.iter().map(|message| message.content.len()).sum::<usize>()
                > self.max_characters
        {
            self.messages.pop_front();
            self.messages.pop_front();
        }
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_whole_exchanges() {
        let mut memory = ConversationMemory::new(2, 10_000);
        memory.add_exchange("u1".into(), "a1".into());
        memory.add_exchange("u2".into(), "a2".into());
        memory.add_exchange("u3".into(), "a3".into());
        let values: Vec<_> = memory.snapshot().into_iter().map(|item| item.content).collect();
        assert_eq!(values, ["u2", "a2", "u3", "a3"]);
    }
}

