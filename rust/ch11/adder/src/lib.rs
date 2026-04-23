#[derive(Debug)]

struct Rectangle{
    width: u32,
    height: u32,
}

impl Rectangle{
    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
}

#[cfg(test)]
mod test{
    use super::*;


    #[test]
    fn larger_can_hold_smaller(){
        let larger = Rectangle{
            width: 8,
            height: 7,
        };
        let smaller = Rectangle{
            width: 5,
            height: 1,
        };

        assert!(larger.can_hold(&smaller));
    }

    #[test]
    fn larger_cannot_hold_smaller(){
        let larger = Rectangle{
            width: 8,
            height: 7,
        };
        let smaller = Rectangle{
            width: 5,
            height: 1,
        };

        assert!(!smaller.can_hold(&larger));
    }
}

pub fn greeting(name: &str) -> String{
    format!("Hello {}!",name)
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn greeting_contains_name(){
        let result = greeting("Carol");
        assert!(result.contains("Carol"),
        "Greeting didn't contain name, value was '{}'",
        result
    );
    }
}


pub struct Guess{
    value: i32
}

impl Guess {
    pub fn new(value: i32) -> Guess{
        if value < 1 {
            panic!("Guess value must be ge 1, got {}.", value);
        }else if value > 100 {
            panic!("Guess value must be le 100, git {}.",value);
        }
        Guess { value }
    }
}

#[cfg(test)]
mod tests2{
    use super::*;

    #[test]
    #[should_panic(expected = "le 100")]
    fn greater_than_100(){
        Guess::new(200);
    }
}

#[cfg(test)]
mod tests3{
    #[test]
    fn it_works() -> Result<(), String>{
        if 2 + 2 == 4{
            Ok(())
        }else {
            Err(String::from("2+2 != 4"))
        }
    }
}

fn prints_and_returns_10(a: i32) -> i32{
    println!("I got the value {}", a);
    10
}

#[cfg(test)]
mod tests4{
    use super::*;

    #[test]
    fn this_test_will_pass(){
        let value = prints_and_returns_10(4);
        assert_eq!(10,value);
    }

    #[test]
    #[ignore]
    fn this_test_will_fail(){
        let value = prints_and_returns_10(8);
        assert_eq!(5,value);
    }
}

pub fn add_two(a: i32) -> i32{
    a + 2
}

fn internal_adder(a: i32, b: i32) -> i32{
    a+b
}

#[cfg(test)]
mod tests5{
    use super::*; 

    #[test]
    fn add_two_and_two(){
        assert_eq!(4, add_two(2));
    }

    #[test]
    fn add_three_and_two(){
        assert_eq!(5,add_two(3));
    }

    #[test]
    fn one_hundred(){
        assert_eq!(102,add_two(100));
    }

    #[test]
    fn internal(){
        assert_eq!(4, internal_adder(2, 2));
    }
}