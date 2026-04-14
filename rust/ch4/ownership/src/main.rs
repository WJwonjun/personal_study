fn main() {
    let my_string = String::from("hello world");

    let word = first_word(&my_string[..6]);
    let word = first_word(&my_string[..]);

    let word = first_word(&my_string);
    
    let my_string_literal = "hello world";

    let word = first_word(&my_string_literal[0..6]);
    let word = first_word(&my_string_literal[..]);

    let word = first_word(my_string_literal);
}

fn takes_ownership(some_string: String){
    println!("{}", some_string);
}

fn makes_copy(some_integer: i32){
    println!("{}",some_integer);
}

fn gives_ownership() -> String{
    let some_string = String::from("yours");
    some_string
}

fn takes_and_gives_back(a_string:String) -> String {
    a_string
}

fn first_word(s: &str) -> &str { 
    let bytes = s.as_bytes();

    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            // 조기 반환: 공백을 찾으면 즉시 슬라이스 반환
            return &s[..i]; 
        } 
    }
    // 마지막 줄: return 없이 세미콜론을 빼서 전체 슬라이스 반환
    &s[..] 
}
