struct User{
    active: bool,
    username: String,
    email: String,
    sing_in_count: u64,
}

fn main() {
    let mut user1 = User{
        active: true,
        username: String::from("someusername123"),
        email: String::from("someone@example.com"),
        sing_in_count: 1,
    };
    user1.email = String::from("anotheremail@example.com");

    let user2 = User{
        email: String::from("someone@example.com"),
        ..user1
    };
}

fn build_user(email:String, username:String) -> User{
    User{
        active: true,
        username,
        email,
        sing_in_count:1,
    }
}