window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        build_response: function(token) {

            const chatHistory = JSON.parse(localStorage.getItem("chatHistory")) || [];
            const last = chatHistory[chatHistory.length - 1];

            if (last && last.role === "assistant") {
                // const updated = chatHistory;
                chatHistory[chatHistory.length - 1] = {
                    ...last,
                    content: last.content + token
                };
                localStorage.setItem("chatHistory", JSON.stringify(chatHistory));

                return chatHistory;
            }
            initAssistantObj = { role: "assistant", content: token }
            chatHistory.push(initAssistantObj)
            localStorage.setItem("chatHistory", JSON.stringify(chatHistory))
            return chatHistory;
        },
        add_user_msg: function(newMsg) {
            const chatHistory = JSON.parse(localStorage.getItem("chatHistory")) || [];
            chatHistory.push(newMsg);
            localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
            return "";
        },
    }
});
