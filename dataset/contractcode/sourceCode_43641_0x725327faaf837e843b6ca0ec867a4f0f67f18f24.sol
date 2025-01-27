/**
 *Submitted for verification at Etherscan.io on 2017-09-07
*/

pragma solidity ^0.4.16;
 
/* 
   Эквивалент белорусского рубля BYN,
  cвободно конвертируемый в любые платежи по Беларуси через ЕРИП
   и в балансы операторов сотовой связи Беларуси.
   1 BYN (Belorussian Rubble) = 1 BYN в системе ЕРИП.
   Комиссии при транзакциях (переводах) данного эквивалента уплачиваются сторонним майнерам Ethereum в валюте ETH, так как основаны на самом надежном блокчейне Ethereum
   
   Система не направлена на получении какой-либо прибыли, действует на некоммерческой основе. 
   Нет комиссий или иных способов получения прибыли пользователями или создателями системы.
   Токен  1 BYN (Belorussian Rubble) не является ни денежным суррогатом, ни ценной бумагой.
  BYN (Belorussian Rubble) - это учетная единица имеющихся у участников системы свободных средств в системе ЕРИП.
  Покупка или продажа токенов системы  BYN (Belorussian Rubble) является покупкой или продажей белорусских рублей в системе ЕРИП. 
  Контракт системы хранится на серверах блокчейна Ethereum и не подлежит изменению, редактированию  ввиду невозможности редактирования истории изменений состояния блокчейна.
  Переводы внутри системы невозможно отменить, вернуть или невозможно закрыть каким-либо участникам права на использование системы.
  У системы нет ни модератора, ни хозяина, создана для благотворительных целей без целей извлечения какой-либо прибыли. Участники системы действуют на добровольной основе, самостоятельно и без необходимости согласия создателями системы. 
  BYN (Belorussian Rubble) является смартконтрактм (скрпитом) и не подлежит регулированию. 
  
 */
contract ERC20Basic {
  uint256 public totalSupply;
  function balanceOf(address who) constant returns (uint256);
  function transfer(address to, uint256 value) returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}
 
/*
   ERC20 interface
  see https://github.com/ethereum/EIPs/issues/20
 */
contract ERC20 is ERC20Basic {
  function allowance(address owner, address spender) constant returns (uint256);
  function transferFrom(address from, address to, uint256 value) returns (bool);
  function approve(address spender, uint256 value) returns (bool);
  event Approval(address indexed owner, address indexed spender, uint256 value);
}
 
/*  SafeMath - the lowest gas library
  Math operations with safety checks that throw on error
 */
library SafeMath {
    
  function mul(uint256 a, uint256 b) internal constant returns (uint256) {
    uint256 c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }
 
  function div(uint256 a, uint256 b) internal constant returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return c;
  }
 
  function sub(uint256 a, uint256 b) internal constant returns (uint256) {
    assert(b <= a);
    return a - b;
  }
 
  function add(uint256 a, uint256 b) internal constant returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
  
}
 
/*
Basic token
 Basic version of StandardToken, with no allowances. 
 */
contract BasicToken is ERC20Basic {
    
  using SafeMath for uint256;
 
  mapping(address => uint256) balances;
 
 function transfer(address _to, uint256 _value) returns (bool) {
    balances[msg.sender] = balances[msg.sender].sub(_value);
    balances[_to] = balances[_to].add(_value);
    Transfer(msg.sender, _to, _value);
    return true;
  }
 
  /*
  Gets the balance of the specified address.
   param _owner The address to query the the balance of. 
   return An uint256 representing the amount owned by the passed address.
  */
  function balanceOf(address _owner) constant returns (uint256 balance) {
    return balances[_owner];
  }
 
}
 
/* Implementation of the basic standard token.
  https://github.com/ethereum/EIPs/issues/20
 */
contract StandardToken is ERC20, BasicToken {
 
  mapping (address => mapping (address => uint256)) allowed;
 
  /*
    Transfer tokens from one address to another
    param _from address The address which you want to send tokens from
    param _to address The address which you want to transfer to
    param _value uint256 the amout of tokens to be transfered
   */
  function transferFrom(address _from, address _to, uint256 _value) returns (bool) {
    var _allowance = allowed[_from][msg.sender];
 
    // Check is not needed because sub(_allowance, _value) will already throw if this condition is not met
    // require (_value <= _allowance);
 
    balances[_to] = balances[_to].add(_value);
    balances[_from] = balances[_from].sub(_value);
    allowed[_from][msg.sender] = _allowance.sub(_value);
    Transfer(_from, _to, _value);
    return true;
  }
 
  /*
  Aprove the passed address to spend the specified amount of tokens on behalf of msg.sender.
   param _spender The address which will spend the funds.
   param _value The amount of Roman Lanskoj's tokens to be spent.
   */
  function approve(address _spender, uint256 _value) returns (bool) {
 
    // To change the approve amount you first have to reduce the addresses`
    //  allowance to zero by calling `approve(_spender, 0)` if it is not
    //  already 0 to mitigate the race condition described here:
    //  https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
    require((_value == 0) || (allowed[msg.sender][_spender] == 0));
 
    allowed[msg.sender][_spender] = _value;
    Approval(msg.sender, _spender, _value);
    return true;
  }
 
  /*
  Function to check the amount of tokens that an owner allowed to a spender.
  param _owner address The address which owns the funds.
  param _spender address The address which will spend the funds.
  return A uint256 specifing the amount of tokens still available for the spender.
   */
  function allowance(address _owner, address _spender) constant returns (uint256 remaining) {
    return allowed[_owner][_spender];
}
}
 
/*
The Ownable contract has an owner address, and provides basic authorization control
 functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
    
  address public owner;
 
 
  function Ownable() {
    owner = msg.sender;
  }
 
  /*
  Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }
 
  /*
  Allows the current owner to transfer control of the contract to a newOwner.
  param newOwner The address to transfer ownership to.
   */
  function transferOwnership(address newOwner) onlyOwner {
    require(newOwner != address(0));      
    owner = newOwner;
  }
 
}
 
contract TheLiquidToken is StandardToken, Ownable {
    // mint can be finished and token become fixed for forever
  event Mint(address indexed to, uint256 amount);
  event MintFinished();
  bool mintingFinished = false;
  modifier canMint() {
    require(!mintingFinished);
    _;
  }
 
 function mint(address _to, uint256 _amount) onlyOwner canMint returns (bool) {
    totalSupply = totalSupply.add(_amount);
    balances[_to] = balances[_to].add(_amount);
    Mint(_to, _amount);
    return true;
  }
 
  /*
  Function to stop minting new tokens.
  return True if the operation was successful.
   */
  function finishMinting() onlyOwner returns (bool) {}
  
  function burn(uint _value)
        public
    {
        require(_value > 0);

        address burner = msg.sender;
        balances[burner] = balances[burner].sub(_value);
        totalSupply = totalSupply.sub(_value);
        Burn(burner, _value);
    }

    event Burn(address indexed burner, uint indexed value);
  
}
    
contract BYN is TheLiquidToken {
  string public constant name = "Belorussian Rubble";
  string public constant symbol = "BYN";
  uint public constant decimals = 2;
  uint256 public initialSupply;
    
  function Fricacoin () { 
     totalSupply = 1200 * 10 ** decimals;
      balances[msg.sender] = totalSupply;
      initialSupply = totalSupply; 
        Transfer(0, this, totalSupply);
        Transfer(this, msg.sender, totalSupply);
  }
}