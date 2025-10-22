function togglePassword(id) {
  let input = document.getElementById(id);
  if (input.type === "password") {
    input.type = "text";
  } else {
    input.type = "password";
  }
}
