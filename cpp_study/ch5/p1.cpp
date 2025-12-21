class string{
    char *str;
    int len;

    public:
        string(char c, int n);
        string(const char* s);
        string(cont string7 s);
        ~string();

    void add_string(const string& s);
    void copy_string(const string& s);
    int strlen();
};
int string::strlen(){
    return len;
}

string::string(char c, int n){
    len = n;
    str = new char[len+1];
    for (int i = 0; i < len; i++) {
		str[i] = c;
	}
	str[len] = '\0';
}

string::string(const char* s) {
	len = strlen(s);
	str = new char[len + 1];
	std::strcpy(str, s);
}

string::string(const string& s) {
	len = s.len;
	str = new char[len + 1];
	std::strcpy(str, s.str);
}

string::~string() {
	delete[] str;
}

void string::add_string(const string& s){
    char* temp = new char[len + s.len + 1];
    std::strcpy(temp, str);
	std::strcat(temp, s.str);

	delete[] str;
	str = temp;
	len += s.len;
}


void string::copy_string(const string& s) {
	delete[] str;
	len = s.len;
	str = new char[len + 1];
	std::strcpy(str, s.str);
}