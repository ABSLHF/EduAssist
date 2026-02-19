(() => {
  const map = {
    "Authorize": "授权",
    "Authorized": "已授权",
    "Logout": "退出授权",
    "Close": "关闭",
    "Scopes are used to grant an application different levels of access to data on behalf of the end user.": "权限范围用于代表最终用户授予应用访问数据的不同级别权限。",
    "Each API may declare one or more scopes.": "每个接口可能声明一个或多个权限范围。",
    "Available authorizations": "可用授权",
    "Each API may declare one or more scopes.": "每个接口可能声明一个或多个权限范围。",
    "API requires the following scopes. Select which ones you want to grant to Swagger UI.": "接口需要以下权限范围。请选择要授予 Swagger UI 的权限。",
    "OAuth2PasswordBearer (OAuth2, password)": "OAuth2PasswordBearer（OAuth2，密码模式）",
    "Token URL:": "令牌地址：",
    "Flow:": "授权方式：",
    "username:": "用户名：",
    "password:": "密码：",
    "client_id:": "客户端ID：",
    "client_secret:": "客户端密钥：",
    "Client credentials location:": "客户端凭据位置：",
    "Authorization header": "Authorization 请求头",
    "Try it out": "试一试",
    "Execute": "执行",
    "Clear": "清除",
    "Cancel": "取消",
    "Parameters": "参数",
    "Request body": "请求体",
    "Responses": "响应",
    "Server response": "服务器响应",
    "Description": "说明",
    "Schemas": "数据模型",
    "Models": "数据模型",
    "Value": "值",
    "Example Value": "示例值",
    "Required": "必填",
    "Optional": "可选",
    "Download": "下载"
  };

  const excluded = new Set(["CODE", "PRE", "SCRIPT", "STYLE"]);

  function translateNode(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = node.nodeValue;
      const trimmed = text.trim();
      if (trimmed && map[trimmed]) {
        node.nodeValue = text.replace(trimmed, map[trimmed]);
      }
      return;
    }

    if (node.nodeType === Node.ELEMENT_NODE) {
      if (excluded.has(node.tagName)) return;
      node.childNodes.forEach(translateNode);
    }
  }

  function translateAll() {
    translateNode(document.body);
  }

  const observer = new MutationObserver(() => translateAll());
  window.addEventListener("load", () => {
    translateAll();
    observer.observe(document.body, { childList: true, subtree: true });
  });
})();
