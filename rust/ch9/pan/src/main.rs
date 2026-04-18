use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error>{
    let username_file = File::open("hello.txt")?;
    /* 
    {
        Ok(file) => file,
        Err(e) => return Err(e),;
    };
    */

    let mut username = String::new();

    username_file.read_to_string(&mut username)?;
    Ok(username)
    /* 
    {
        Ok(_) => Ok(username),
        Err(e) => Err(e)
    }
    */
}


fn main() {
    let greeting_file_result = File::open("hello.txt").unwrap_or_else(|error|{
        if error.kind() == ErrorKind::NotFound{
            File::create("hello.txt").unwrap_or_else(|error|{
                panic!("Problem creating the file: {:?}", error);
            })
        }else {
            panic!("Problem opening the file: {:?}",error);
        }
    });
}
